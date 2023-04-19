import torch
import numpy as np
import random


def calc_metrics(prediction_trusted, test_head_dict, head_ids, tail_ids, neg_rate):
    """
    prediction_scores: (n_heads, n_tails)
    """
    tail_ids = list(tail_ids)
    ground_trusted = torch.zeros(
        [len(head_ids), len(tail_ids)], dtype=torch.int32)

    y_pred = []
    y_true = []

    for idx, h_id in enumerate(head_ids):
        test_pos_tail_list = []
        test_neg_tail_list = []

        if h_id in test_head_dict:
            test_pos_tail_list = test_head_dict[h_id]

        while True:
            if len(test_neg_tail_list) == len(test_pos_tail_list)*neg_rate:
                break

            neg_tail_id = random.choice(list(tail_ids))
            if neg_tail_id not in test_pos_tail_list and neg_tail_id not in test_neg_tail_list:
                test_neg_tail_list.append(neg_tail_id)

        ground_trusted[idx][[tail_ids.index(x) for x in test_pos_tail_list]] = 1
        # prediction_trusted[idx][[tail_ids.index(x) for x in test_pos_tail_list]] = 1

        test_tails = test_pos_tail_list + test_neg_tail_list

        y_pred.append(prediction_trusted[idx][[tail_ids.index(x) for x in test_tails]])
        y_true.append(ground_trusted[idx][[tail_ids.index(x) for x in test_tails]])


    y_pred = torch.cat(y_pred, 0)

    y_true = torch.cat(y_true, 0)

    metrics_dict = {}

    # accuracy_tensor = sum(y_pred == y_true) / len(y_true)
    # accuracy = float(sum(accuracy_tensor)/len(accuracy_tensor))
    accuracy = calc_accuracy(y_pred=y_pred, y_true=y_true)
    metrics_dict['accuracy'] = accuracy
    # precision tp / (tp + fp)
    precision = calc_precision(y_pred=y_pred, y_true=y_true)
    metrics_dict['precision'] = precision
    # recall: tp / (tp + fn)
    recall = calc_recall(y_pred=y_pred, y_true=y_true)
    metrics_dict['recall'] = recall
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = calc_f1(precision, recall)
    metrics_dict['f1'] = f1
    return metrics_dict

def calc_metrics(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    
    metrics_dict = {}

    print(y_pred)
    print("================================================")
    print(y_true)
    # accuracy_tensor = sum(y_pred == y_true) / len(y_true)
    # accuracy = float(sum(accuracy_tensor)/len(accuracy_tensor))
    accuracy = calc_accuracy(y_pred=y_pred, y_true=y_true)
    metrics_dict['accuracy'] = accuracy
    # precision tp / (tp + fp)
    precision = calc_precision(y_pred=y_pred, y_true=y_true)
    metrics_dict['precision'] = precision
    # recall: tp / (tp + fn)
    recall = calc_recall(y_pred=y_pred, y_true=y_true)
    metrics_dict['recall'] = recall
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = calc_f1(precision, recall)
    metrics_dict['f1'] = f1
    return metrics_dict


def calc_accuracy(y_pred=None, y_true=None):
    num_correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            num_correct += 1
    return num_correct / len(y_true)

def calc_recall(y_pred=None, y_true=None):
    TP = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            FN += 1

    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def calc_precision(y_pred=None, y_true=None):
    TP = 0
    FP = 0
    for i in range(len(y_pred)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def calc_f1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

