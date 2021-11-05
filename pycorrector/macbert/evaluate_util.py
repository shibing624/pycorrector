# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""


def compute_corrector_prf(results, logger):
    """
    copy from https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py
    """
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    for item in results:
        src, tgt, predict = item
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if detection_precision + detection_recall == 0:
        detection_f1 = 0
    else:
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
    logger.info(
        "The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall,
                                                                           detection_f1))

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we wanna to see the precision improve by using the confusionset
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results[i][2][j])
                if results[i][1][j] == results[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if results[i][1][j] in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if correction_precision + correction_recall == 0:
        correction_f1 = 0
    else:
        correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall)
    logger.info("The correction result is precision={}, recall={} and F1={}".format(correction_precision,
                                                                                    correction_recall,
                                                                                    correction_f1))

    return detection_f1, correction_f1


def compute_sentence_level_prf(results, logger):
    """
    自定义的句级prf，设定需要纠错为正样本，无需纠错为负样本
    :param results:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = len(results)

    for item in results:
        src, tgt, predict = item

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == predict:
                TN += 1
            # 预测为正
            else:
                FP += 1
        # 正样本
        else:
            # 预测也为正
            if tgt == predict:
                TP += 1
            # 预测为负
            else:
                FN += 1

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    logger.info(f'Sentence Level: acc:{acc:.6f}, precision:{precision:.6f}, recall:{recall:.6f}, f1:{f1:.6f}')
    return acc, precision, recall, f1


def report_prf(tp, fp, fn, phase, logger=None, return_dict=False):
    # For the detection Precision, Recall and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    if phase and logger:
        logger.info(f"The {phase} result is: "
                    f"{precision:.4f}/{recall:.4f}/{f1_score:.4f} -->\n"
                    # f"precision={precision:.6f}, recall={recall:.6f} and F1={f1_score:.6f}\n"
                    f"support: TP={tp}, FP={fp}, FN={fn}")
    if return_dict:
        ret_dict = {
            f'{phase}_p': precision,
            f'{phase}_r': recall,
            f'{phase}_f1': f1_score}
        return ret_dict
    return precision, recall, f1_score


def compute_corrector_prf_faspell(results, logger=None, strict=True):
    """
    All-in-one measure function.
    based on FASpell's measure script.
    :param results: a list of (wrong, correct, predict, ...)
    both token_ids or characters are fine for the script.
    :param logger: take which logger to print logs.
    :param strict: a more strict evaluation mode (all-char-detected/corrected)
    References:
        sentence-level PRF: https://github.com/iqiyi/
        FASPell/blob/master/faspell.py
    """

    corrected_char, wrong_char = 0, 0
    corrected_sent, wrong_sent = 0, 0
    true_corrected_char = 0
    true_corrected_sent = 0
    true_detected_char = 0
    true_detected_sent = 0
    accurate_detected_sent = 0
    accurate_corrected_sent = 0
    all_sent = 0

    for item in results:
        # wrong, correct, predict, d_tgt, d_predict = item
        wrong, correct, predict = item[:3]

        all_sent += 1
        wrong_num = 0
        corrected_num = 0
        original_wrong_num = 0
        true_detected_char_in_sentence = 0

        for c, w, p in zip(correct, wrong, predict):
            if c != p:
                wrong_num += 1
            if w != p:
                corrected_num += 1
                if c == p:
                    true_corrected_char += 1
                if w != c:
                    true_detected_char += 1
                    true_detected_char_in_sentence += 1
            if c != w:
                original_wrong_num += 1

        corrected_char += corrected_num
        wrong_char += original_wrong_num
        if original_wrong_num != 0:
            wrong_sent += 1
        if corrected_num != 0 and wrong_num == 0:
            true_corrected_sent += 1

        if corrected_num != 0:
            corrected_sent += 1

        if strict:  # find out all faulty wordings' potisions
            true_detected_flag = (true_detected_char_in_sentence == original_wrong_num \
                                  and original_wrong_num != 0 \
                                  and corrected_num == true_detected_char_in_sentence)
        else:  # think it has faulty wordings
            true_detected_flag = (corrected_num != 0 and original_wrong_num != 0)

        # if corrected_num != 0 and original_wrong_num != 0:
        if true_detected_flag:
            true_detected_sent += 1
        if correct == predict:
            accurate_corrected_sent += 1
        if correct == predict or true_detected_flag:
            accurate_detected_sent += 1

    counts = {  # TP, FP, TN for each level
        'det_char_counts': [true_detected_char,
                            corrected_char - true_detected_char,
                            wrong_char - true_detected_char],
        'cor_char_counts': [true_corrected_char,
                            corrected_char - true_corrected_char,
                            wrong_char - true_corrected_char],
        'det_sent_counts': [true_detected_sent,
                            corrected_sent - true_detected_sent,
                            wrong_sent - true_detected_sent],
        'cor_sent_counts': [true_corrected_sent,
                            corrected_sent - true_corrected_sent,
                            wrong_sent - true_corrected_sent],
        'det_sent_acc': accurate_detected_sent / all_sent,
        'cor_sent_acc': accurate_corrected_sent / all_sent,
        'all_sent_count': all_sent,
    }

    details = {}
    for phase in ['det_char', 'cor_char', 'det_sent', 'cor_sent']:
        dic = report_prf(
            *counts[f'{phase}_counts'],
            phase=phase, logger=logger,
            return_dict=True)
        details.update(dic)
    details.update(counts)
    return details
