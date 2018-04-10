# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os
from utils.io_utils import get_logger

logger = get_logger(__name__)


def eval(preds, label_ids_dict, ids_label_dict, X_test, sid_test, out_path=None):
    with open(out_path, 'w') as f:
        for i in range(len(X_test)):
            sent = X_test[i]
            sid = sid_test[i]

            label = []
            for j in range(len(sent)):
                if sent[j] != 0:
                    label.append(preds[i][j])

            error_flag = False
            is_correct = False

            current_error = 0
            start_pos = 0
            end_pos = 0
            for k in range(len(label)):
                if (label[k] == label_ids_dict['R'] or label[k] == label_ids_dict['S'] or \
                                label[k] == label_ids_dict['M'] or label[k] == label_ids_dict[
                    'W']) and error_flag == False:
                    error_flag = True
                    start_pos = k + 1
                    current_error = label[k]
                    is_correct = True

                if error_flag == True and label[k] != current_error and (
                                        label[k] != label_ids_dict['R'] and label[k] != label_ids_dict['S'] and \
                                        label[k] != label_ids_dict['M'] and label[k] != label_ids_dict['W']):
                    end_pos = k
                    f.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, ids_label_dict[current_error]))

                    error_flag = False
                    current_error = 0

                if error_flag == True and label[k] != current_error and (
                                        label[k] == label_ids_dict['R'] or label[k] == label_ids_dict['S'] or \
                                        label[k] == label_ids_dict['M'] or label[k] == label_ids_dict['W']):
                    end_pos = k
                    f.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, ids_label_dict[current_error]))

                    start_pos = k + 1
                    current_error = label[k]

            if is_correct == False:
                f.write('%s, correct\n' % (sid))

            if i % 100 == 0:
                logger.info('processed: %d/%d' % (i, len(sid_test)))


if __name__ == '__main__':
    pred_out_path = os.path.join('result', 'cged16_hsk_singlelabel_random_bilstm_crf.txt')
