# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Conv Seq2Seq model
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from pycorrector.seq2seq.conv_seq2seq_utils import (
    gen_examples, read_vocab, create_dataset,
    one_hot, save_word_dict, load_word_dict,
    SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
)

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim,
            emb_dim=256,
            hid_dim=512,
            n_layers=2,
            kernel_size=3,
            dropout=0.25,
            device=torch.device('cuda'),
            max_length=128
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [0, 1, 2, 3, ..., src len - 1]
        # pos = [batch size, src len]
        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        # tok_embedded = pos_embedded = [batch size, src len, emb dim]
        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        # embedded = [batch size, src len, emb dim]
        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        # conv_input = [batch size, src len, hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, src len]
        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2 * hid dim, src len]
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, src len]
            # apply residual connection
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, src len]
            # set conv_input to conved for next loop iteration
            conv_input = conved
        # end convolutional blocks
        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        # conved = [batch size, src len, emb dim]
        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        # combined = [batch size, src len, emb dim]
        return conved, combined


class Decoder(nn.Module):
    def __init__(
            self,
            output_dim,
            emb_dim=256,
            hid_dim=512,
            n_layers=2,
            kernel_size=3,
            dropout=0.25,
            trg_pad_idx=0,
            device=torch.device('cuda'),
            max_length=128
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        """
        Attention
        :param embedded: embedded = [batch size, trg len, emb dim]
        :param conved: conved = [batch size, hid dim, trg len]
        :param encoder_conved: encoder_conved = encoder_combined = [batch size, src len, emb dim]
        :param encoder_combined: permute and convert back to emb dim
        :return:
        """
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        # conved_emb = [batch size, trg len, emb dim]
        combined = (conved_emb + embedded) * self.scale
        # combined = [batch size, trg len, emb dim]
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        # energy = [batch size, trg len, src len]
        attention = F.softmax(energy, dim=2)
        # attention = [batch size, trg len, src len]
        attended_encoding = torch.matmul(attention, encoder_combined)
        # attended_encoding = [batch size, trg len, emd dim]
        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)
        # attended_encoding = [batch size, trg len, hid dim]
        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        # attended_combined = [batch size, hid dim, trg len]
        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        """
        Get output and attention
        :param trg: trg = [batch size, trg len]
        :param encoder_conved: encoder_conved = encoder_combined = [batch size, src len, emb dim]
        :param encoder_combined:
        :return:
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        # create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, trg len]
        # embed tokens and positions
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]
        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        # embedded = [batch size, trg len, emb dim]
        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)
        # conv_input = [batch size, trg len, hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, trg len]
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)
            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)
            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
            # pass through convolutional layer
            conved = conv(padded_conv_input)
            # conved = [batch size, 2 * hid dim, trg len]
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, trg len]
            # calculate attention
            attention, conved = self.calculate_attention(embedded,
                                                         conved,
                                                         encoder_conved,
                                                         encoder_combined)
            # attention = [batch size, trg len, src len]
            # apply residual connection
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, trg len]
            # set conv_input to conved for next loop iteration
            conv_input = conved
        conved = self.hid2emb(conved.permute(0, 2, 1))
        # conved = [batch size, trg len, emb dim]
        output = self.fc_out(self.dropout(conved))
        # output = [batch size, trg len, output dim]
        return output, attention


class ConvSeq2Seq(nn.Module):
    def __init__(
            self,
            encoder_vocab_size,
            decoder_vocab_size,
            embed_size,
            enc_hidden_size,
            dec_hidden_size,
            dropout=0.25,
            trg_pad_idx=0,
            device=device,
            max_length=128
    ):
        super().__init__()
        self.encoder = Encoder(input_dim=encoder_vocab_size,
                               emb_dim=embed_size,
                               hid_dim=enc_hidden_size,
                               n_layers=2,
                               kernel_size=3,
                               dropout=dropout,
                               device=device,
                               max_length=max_length)
        self.decoder = Decoder(output_dim=decoder_vocab_size,
                               emb_dim=embed_size,
                               hid_dim=dec_hidden_size,
                               n_layers=2,
                               kernel_size=3,
                               dropout=dropout,
                               trg_pad_idx=trg_pad_idx,
                               device=device,
                               max_length=max_length)
        self.max_length = max_length
        self.device = device

    def forward(self, src, trg):
        """
        Calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        :param src:src = [batch size, src len]
        :param trg: trg = [batch size, trg len - 1] (<eos> token sliced off the end)
        :return:
        """
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)
        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]
        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)
        # output = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]
        return output, attention

    def translate(self, x, sos):
        """
        Predict x
        :param x: input tensor
        :param sos: SOS tensor
        :return: preds, attns
        """
        encoder_conved, encoder_combined = self.encoder(x)
        preds = []
        attns = []
        trg_indexes = [sos]
        for i in range(self.max_length):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
            output, attention = self.decoder(trg_tensor, encoder_conved, encoder_combined)
            pred = output.argmax(2)[:, -1].item()
            preds.append(pred)
            attns.append(attention)
            trg_indexes.append(pred)

        return preds, attns


class ConvSeq2SeqModel:
    def __init__(
            self,
            embed_size=128,
            hidden_size=128,
            dropout=0.25,
            num_epochs=10,
            batch_size=32,
            model_dir="outputs/",
            max_length=128,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.max_length = max_length
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.model = None
        self.model_path = os.path.join(self.model_dir, 'convseq2seq.pth')
        logger.debug(f"Device: {device}")
        self.loss_fn = nn.CrossEntropyLoss()
        self.src_vocab_path = os.path.join(self.model_dir, "vocab_source.txt")
        self.trg_vocab_path = os.path.join(self.model_dir, "vocab_target.txt")
        if os.path.exists(self.src_vocab_path):
            self.src_2_ids = load_word_dict(self.src_vocab_path)
            self.trg_2_ids = load_word_dict(self.trg_vocab_path)
            self.id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
            self.trg_pad_idx = self.trg_2_ids[PAD_TOKEN]
        else:
            self.src_2_ids = None
            self.trg_2_ids = None
            self.id_2_trgs = None
            self.trg_pad_idx = None

    def train_model(self, train_data, eval_data=None):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence
                        If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
        Returns:
            training_details: training loss 
        """  # noqa: ignore flake8"

        logger.info("Training model...")
        os.makedirs(self.model_dir, exist_ok=True)
        source_texts, target_texts = create_dataset(train_data)

        self.src_2_ids = read_vocab(source_texts)
        self.trg_2_ids = read_vocab(target_texts)
        self.trg_pad_idx = self.trg_2_ids[PAD_TOKEN]
        save_word_dict(self.src_2_ids, self.src_vocab_path)
        save_word_dict(self.trg_2_ids, self.trg_vocab_path)
        train_src, train_trg = one_hot(source_texts, target_texts, self.src_2_ids, self.trg_2_ids, sort_by_len=True)

        id_2_srcs = {v: k for k, v in self.src_2_ids.items()}
        id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
        logger.debug(f'train src: {[id_2_srcs[i] for i in train_src[0]]}')
        logger.debug(f'train trg: {[id_2_trgs[i] for i in train_trg[0]]}')

        self.model = ConvSeq2Seq(
            encoder_vocab_size=len(self.src_2_ids),
            decoder_vocab_size=len(self.trg_2_ids),
            embed_size=self.embed_size,
            enc_hidden_size=self.hidden_size,
            dec_hidden_size=self.hidden_size,
            dropout=self.dropout,
            trg_pad_idx=self.trg_pad_idx,
            device=device,
            max_length=self.max_length
        )
        self.model.to(device)
        logger.debug(self.model)
        optimizer = torch.optim.Adam(self.model.parameters())

        train_data = gen_examples(train_src, train_trg, self.batch_size, self.max_length)
        train_losses = []
        best_loss = 1e3
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.
            total_iter = 0.
            for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(train_data):
                src = torch.from_numpy(mb_x).to(device).long()
                trg = torch.from_numpy(mb_y).to(device).long()
                output, attn = self.model(src, trg[:, :-1])

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]
                loss = self.loss_fn(output, trg)
                total_loss += loss.item()
                total_iter += 1

                # update optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it % 100 == 0:
                    logger.debug("Epoch :{}/{}, iteration :{}/{} loss:{:.4f}".format(epoch, self.num_epochs,
                                                                                     it, len(train_data),
                                                                                     loss.item()))
            cur_loss = total_loss / total_iter
            train_losses.append(cur_loss)
            logger.debug("Epoch :{}/{}, Training loss:{:.4f}".format(epoch, self.num_epochs, cur_loss))
            if epoch % 1 == 0:
                # find best model
                is_best = cur_loss < best_loss
                best_loss = min(cur_loss, best_loss)
                if is_best:
                    self.save_model()
                    logger.info('Epoch:{}, save new bert model:{}'.format(epoch, self.model_path))
                if eval_data:
                    self.eval_model(eval_data)

        return train_losses

    def eval_model(self, eval_data):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence.
                        If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
        Returns:
            results: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"
        os.makedirs(self.model_dir, exist_ok=True)
        source_texts, target_texts = create_dataset(eval_data)
        logger.info("Evaluating the model...")
        logger.info("Number of examples: {}".format(len(source_texts)))

        if self.src_2_ids is None:
            self.src_2_ids = load_word_dict(self.src_vocab_path)
            self.trg_2_ids = load_word_dict(self.trg_vocab_path)
            self.trg_pad_idx = self.trg_2_ids[PAD_TOKEN]
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = ConvSeq2Seq(
                    encoder_vocab_size=len(self.src_2_ids),
                    decoder_vocab_size=len(self.trg_2_ids),
                    embed_size=self.embed_size,
                    enc_hidden_size=self.hidden_size,
                    dec_hidden_size=self.hidden_size,
                    dropout=self.dropout,
                    trg_pad_idx=self.trg_pad_idx,
                    device=device,
                    max_length=self.max_length
                )
                self.load_model()
                self.model.to(device)
            else:
                raise ValueError("Model not found at {}".format(self.model_path))
        self.model.eval()

        train_src, train_trg = one_hot(source_texts, target_texts, self.src_2_ids, self.trg_2_ids, sort_by_len=True)
        id_2_srcs = {v: k for k, v in self.src_2_ids.items()}
        id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
        logger.debug(f'evaluate src: {[id_2_srcs[i] for i in train_src[0]]}')
        logger.debug(f'evaluate trg: {[id_2_trgs[i] for i in train_trg[0]]}')
        eval_data = gen_examples(train_src, train_trg, self.batch_size, self.max_length)

        loss = 0.
        with torch.no_grad():
            for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(eval_data):
                src = torch.from_numpy(mb_x).to(device).long()
                trg = torch.from_numpy(mb_y).to(device).long()
                output, attn = self.model(src, trg[:, :-1])
                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]
                loss = self.loss_fn(output, trg)
                loss = loss.item()
        logger.info(f"Evaluation loss: {loss}")
        return {'loss': loss}

    def predict(self, sentences, silent=True):
        """
        Performs predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. 
            silent: A boolean flag to indicate whether to log the progress to stdout.
        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        if self.src_2_ids is None:
            self.src_2_ids = load_word_dict(self.src_vocab_path)
            self.trg_2_ids = load_word_dict(self.trg_vocab_path)
            self.trg_pad_idx = self.trg_2_ids[PAD_TOKEN]
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = ConvSeq2Seq(
                    encoder_vocab_size=len(self.src_2_ids),
                    decoder_vocab_size=len(self.trg_2_ids),
                    embed_size=self.embed_size,
                    enc_hidden_size=self.hidden_size,
                    dec_hidden_size=self.hidden_size,
                    dropout=self.dropout,
                    trg_pad_idx=self.trg_pad_idx,
                    device=device,
                    max_length=self.max_length
                )
                self.load_model()
                self.model.to(device)
            else:
                raise ValueError("Model not found at {}".format(self.model_path))
        self.model.eval()
        result = []
        for query in tqdm(sentences, desc="Generating outputs", disable=silent):
            out = []
            tokens = [token.lower() for token in query]
            tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
            src_ids = [self.src_2_ids[i] for i in tokens if i in self.src_2_ids]
            sos_idx = self.trg_2_ids[SOS_TOKEN]

            src_tensor = torch.from_numpy(np.array(src_ids).reshape(1, -1)).long().to(device)
            translation, attn = self.model.translate(src_tensor, sos_idx)
            translation = [self.id_2_trgs[i] for i in translation if i in self.id_2_trgs]
            for word in translation:
                if word != EOS_TOKEN:
                    out.append(word)
                else:
                    break
            result.append(''.join(out))
        return result

    def save_model(self):
        logger.info(f"Saving model into {self.model_path}")
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
