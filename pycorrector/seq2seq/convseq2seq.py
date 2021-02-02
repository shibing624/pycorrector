# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Conv Seq2Seq model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim=256,
                 hid_dim=512,
                 n_layers=2,
                 kernel_size=3,
                 dropout=0.25,
                 device=torch.device('cuda'),
                 max_length=128):
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
    def __init__(self,
                 output_dim,
                 emb_dim=256,
                 hid_dim=512,
                 n_layers=2,
                 kernel_size=3,
                 dropout=0.25,
                 trg_pad_idx=0,
                 device=torch.device('cuda'),
                 max_length=128):
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
    def __init__(self,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 embed_size,
                 enc_hidden_size,
                 dec_hidden_size,
                 dropout,
                 trg_pad_idx,
                 device,
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