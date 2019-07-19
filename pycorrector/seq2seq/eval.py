# -*- coding: utf-8 -*-
# Author: Tian Shi <tshi@vt.edu>, XuMing <xuming624@qq.com>
# Brief: evaluate model with val file

import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable

sys.path.append('../..')
from pycorrector.seq2seq.data_reader import create_batch_file, process_minibatch_explicit, \
    show_progress, PAD_TOKEN


def eval(model, model_path, val_path, output_dir, batch_size, vocab2id, src_seq_lens, trg_seq_lens, device):
    model.eval()
    with torch.no_grad():
        losses = []
        start_time = time.time()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            print("error, model file not found.", model_path)
            return
        if not os.path.exists(val_path):
            print("error, val file not found.", val_path)
            return
        val_batch = create_batch_file(output_dir=output_dir,
                                      file_type='validate',
                                      file_path=val_path,
                                      batch_size=batch_size)
        print('The number of batches (test): {}'.format(val_batch))
        for batch_id in range(batch_size):
            ext_id2oov, src_arr, trg_input_arr, src_arr_ex, trg_output_arr_ex = process_minibatch_explicit(
                batch_id=batch_id,
                output_dir=output_dir,
                file_type='validate',
                batch_size=batch_size,
                vocab2id=vocab2id,
                max_lens=[src_seq_lens, trg_seq_lens])
            src_var = Variable(torch.LongTensor(src_arr))
            trg_input_var = Variable(torch.LongTensor(trg_input_arr))
            # extend oov
            src_var_ex = Variable(torch.LongTensor(src_arr_ex))
            trg_output_var_ex = Variable(torch.LongTensor(trg_output_arr_ex))

            src_var = src_var.to(device)
            trg_input_var = trg_input_var.to(device)
            src_var_ex = src_var_ex.to(device)
            trg_output_var_ex = trg_output_var_ex.to(device)

            weight_mask = torch.ones(len(vocab2id) + len(ext_id2oov)).to(device)
            weight_mask[vocab2id[PAD_TOKEN]] = 0
            loss_criterion = torch.nn.NLLLoss(weight=weight_mask).to(device)

            logits, attn_, p_gen, loss_cv = model(src_var.to(device), trg_input_var.to(device))
            logits = torch.softmax(logits, dim=2)
            # use the pointer generator loss
            if len(ext_id2oov) > 0:
                logits = model.cal_dist_explicit(src_var_ex, logits, attn_, p_gen, vocab2id, ext_id2oov)
                logits = logits + 1e-20
            else:
                logits = model.cal_dist(src_var, logits, attn_, p_gen, vocab2id)

            logits = torch.log(logits)
            loss = loss_criterion(
                logits.contiguous().view(-1, len(vocab2id) + len(ext_id2oov)),
                trg_output_var_ex.view(-1))

            losses.append(loss.data.cpu().numpy())
            show_progress(batch_id + 1, batch_size)
            del logits, attn_, p_gen, loss_cv, loss
        print()
        end_time = time.time()
        losses_out = np.average(losses)
        print('model={}, loss={}, time={}'.format(model_path, losses_out, end_time - start_time))
