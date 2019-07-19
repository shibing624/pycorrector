# -*- coding: utf-8 -*-
# Author: Tian Shi <tshi@vt.edu>, XuMing <xuming624@qq.com>
# Brief: Fast beam search

import torch
from torch.autograd import Variable


def tensor_transformer(input_seq, batch_size, beam_size):
    seq = input_seq.unsqueeze(2)
    seq = seq.repeat(1, 1, beam_size, 1)
    seq = seq.contiguous().view(batch_size, beam_size * beam_size, seq.size(3))
    return seq


def fast_beam_search(
        model,
        src_text,
        src_text_ex,
        vocab2id,
        ext_id2oov,
        beam_size=4,
        max_len=20,
        network='lstm',
        pointer_net=True,
        oov_explicit=True,
        attn_decoder=True,
        device=torch.device("cpu")):
    """
    fast beam search
    :param model:
    :param src_text:
    :param src_text_ex:
    :param vocab2id:
    :param ext_id2oov:
    :param beam_size:
    :param max_len:
    :param network:
    :param pointer_net:
    :param oov_explicit:
    :param attn_decoder:
    :return:
    """
    batch_size = src_text.size(0)
    src_seq_len = src_text.size(1)
    src_text_rep = src_text.unsqueeze(1).clone().repeat(1, beam_size, 1).view(-1, src_text.size(1)).to(device)
    if oov_explicit:
        src_text_rep_ex = src_text_ex.unsqueeze(1).clone().repeat(1, beam_size, 1).view(-1, src_text_ex.size(1)).to(
            device)
    if network == 'lstm':
        encoder_hy, (h0_new, c0_new), h_attn_new, past_attn_new, past_dehy_new = model.forward_encoder(src_text_rep)
    else:
        encoder_hy, hidden_decoder_new, h_attn_new, past_attn_new, past_dehy_new = model.forward_encoder(src_text_rep)

    beam_seq = Variable(torch.LongTensor(batch_size, beam_size, max_len + 1).fill_(vocab2id['<pad>'])).to(device)
    beam_seq[:, :, 0] = vocab2id['<s>']
    beam_prb = torch.FloatTensor(batch_size, beam_size).fill_(1.0)
    last_wd = Variable(torch.LongTensor(batch_size, beam_size, 1).fill_(vocab2id['<s>'])).to(device)
    beam_attn_out = Variable(torch.FloatTensor(max_len, batch_size, beam_size, src_seq_len).fill_(0.0)).to(device)

    for j in range(max_len):
        if oov_explicit:
            last_wd[last_wd >= len(vocab2id)] = vocab2id['<unk>']
        if network == 'lstm':
            logits, (h0, c0), h_attn, past_attn, p_gen, attn_, past_dehy = model.forward_onestep_decoder(
                j, last_wd.view(-1, 1), (h0_new, c0_new),
                h_attn_new, encoder_hy, past_attn_new, past_dehy_new)
        else:
            logits, hidden_decoder, h_attn, past_attn, p_gen, attn_, past_dehy = model.forward_onestep_decoder(
                j, last_wd.view(-1, 1), hidden_decoder_new,
                h_attn_new, encoder_hy, past_attn_new, past_dehy_new)
        logits = torch.softmax(logits, dim=2)
        if pointer_net:
            if oov_explicit and len(ext_id2oov) > 0:
                logits = model.cal_dist_explicit(src_text_rep_ex, logits, attn_, p_gen, vocab2id, ext_id2oov)
            else:
                logits = model.cal_dist(src_text_rep, logits, attn_, p_gen, vocab2id)

        prob, wds = logits.data.topk(k=beam_size)
        prob = prob.view(batch_size, beam_size, prob.size(1), prob.size(2))
        wds = wds.view(batch_size, beam_size, wds.size(1), wds.size(2))
        if j == 0:
            beam_prb = prob[:, 0, 0]
            beam_seq[:, :, 1] = wds[:, 0, 0]
            last_wd = Variable(wds[:, 0, 0].unsqueeze(2).clone()).to(device)

            if network == 'lstm':
                h0_new = h0
                c0_new = c0
            else:
                hidden_decoder_new = hidden_decoder
            h_attn_new = h_attn
            attn_new = attn_
            past_attn_new = past_attn
            past_dehy_new = past_dehy
            beam_attn_out[j] = attn_new.view(batch_size, beam_size, attn_new.size(-1))
            continue

        cand_seq = tensor_transformer(beam_seq, batch_size, beam_size)
        cand_seq[:, :, j + 1] = wds.squeeze(2).view(batch_size, -1)
        cand_last_wd = wds.squeeze(2).view(batch_size, -1)

        cand_prob = beam_prb.unsqueeze(1).repeat(1, beam_size, 1).transpose(1, 2)
        cand_prob *= prob[:, :, 0]
        cand_prob = cand_prob.contiguous().view(batch_size, beam_size * beam_size)
        if network == 'lstm':
            h0_new = Variable(torch.zeros(batch_size, beam_size, h0.size(-1))).to(device)
            c0_new = Variable(torch.zeros(batch_size, beam_size, c0.size(-1))).to(device)
        else:
            hidden_decoder_new = Variable(torch.zeros(batch_size, beam_size, hidden_decoder.size(-1))).to(device)
        h_attn_new = Variable(torch.zeros(batch_size, beam_size, h_attn.size(-1))).to(device)
        attn_new = Variable(torch.zeros(batch_size, beam_size, attn_.size(-1))).to(device)
        past_attn_new = Variable(torch.zeros(batch_size, beam_size, past_attn.size(-1))).to(device)
        if attn_decoder:
            pdn_size1, pdn_size2 = past_dehy.size(-2), past_dehy.size(-1)
            past_dehy_new = Variable(torch.zeros(batch_size, beam_size, pdn_size1 * pdn_size2)).to(device)
        if network == 'lstm':
            h0 = h0.view(batch_size, beam_size, h0.size(-1))
            h0 = tensor_transformer(h0, batch_size, beam_size)
            c0 = c0.view(batch_size, beam_size, c0.size(-1))
            c0 = tensor_transformer(c0, batch_size, beam_size)
        else:
            hidden_decoder = hidden_decoder.view(batch_size, beam_size, hidden_decoder.size(-1))
            hidden_decoder = tensor_transformer(hidden_decoder, batch_size, beam_size)
        h_attn = h_attn.view(batch_size, beam_size, h_attn.size(-1))
        h_attn = tensor_transformer(h_attn, batch_size, beam_size)
        attn_ = attn_.view(batch_size, beam_size, attn_.size(-1))
        attn_ = tensor_transformer(attn_, batch_size, beam_size)
        past_attn = past_attn.view(batch_size, beam_size, past_attn.size(-1))
        past_attn = tensor_transformer(past_attn, batch_size, beam_size)
        if attn_decoder:
            past_dehy = past_dehy.contiguous().view(batch_size, beam_size, past_dehy.size(-2) * past_dehy.size(-1))
            past_dehy = tensor_transformer(past_dehy, batch_size, beam_size)
        tmp_prb, tmp_idx = cand_prob.topk(k=beam_size, dim=1)
        for x in range(batch_size):
            for b in range(beam_size):
                last_wd[x, b] = cand_last_wd[x, tmp_idx[x, b]]
                beam_seq[x, b] = cand_seq[x, tmp_idx[x, b]]
                beam_prb[x, b] = tmp_prb[x, b]

                if network == 'lstm':
                    h0_new[x, b] = h0[x, tmp_idx[x, b]]
                    c0_new[x, b] = c0[x, tmp_idx[x, b]]
                else:
                    hidden_decoder_new[x, b] = hidden_decoder[x, tmp_idx[x, b]]
                h_attn_new[x, b] = h_attn[x, tmp_idx[x, b]]
                attn_new[x, b] = attn_[x, tmp_idx[x, b]]
                past_attn_new[x, b] = past_attn[x, tmp_idx[x, b]]
                if attn_decoder:
                    past_dehy_new[x, b] = past_dehy[x, tmp_idx[x, b]]

        beam_attn_out[j] = attn_new
        if network == 'lstm':
            h0_new = h0_new.view(-1, h0_new.size(-1))
            c0_new = c0_new.view(-1, c0_new.size(-1))
        else:
            hidden_decoder_new = hidden_decoder_new.view(-1, hidden_decoder_new.size(-1))
        h_attn_new = h_attn_new.view(-1, h_attn_new.size(-1))
        attn_new = attn_new.view(-1, attn_new.size(-1))
        past_attn_new = past_attn_new.view(-1, past_attn_new.size(-1))
        if attn_decoder:
            past_dehy_new = past_dehy_new.view(-1, pdn_size1, pdn_size2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return beam_seq, beam_prb, beam_attn_out
