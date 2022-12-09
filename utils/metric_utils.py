import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random

# def calc_recall(rank, ground_truth, k):
#     """
#     calculate recall of one example
#     """
#     return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


# def precision_at_k(hit, k):
#     """
#     calculate Precision@k
#     hit: list, element is binary (0 / 1)
#     """
#     hit = np.asarray(hit)[:k]
#     return np.mean(hit)


# def precision_at_k_batch(hits, k):
#     """
#     calculate Precision@k
#     hits: array, element is binary (0 / 1), 2-dim
#     """
#     res = hits[:, :k].mean(axis=1)
#     return res


# def average_precision(hit, cut):
#     """
#     calculate average precision (area under PR curve)
#     hit: list, element is binary (0 / 1)
#     """
#     hit = np.asarray(hit)
#     precisions = [precision_at_k(hit, k + 1)
#                   for k in range(cut) if len(hit) >= k]
#     if not precisions:
#         return 0.
#     return np.sum(precisions) / float(min(cut, np.sum(hit)))


# def dcg_at_k(rel, k):
#     """
#     calculate discounted cumulative gain (dcg)
#     rel: list, element is positive real values, can be binary
#     """
#     rel = np.asfarray(rel)[:k]
#     dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
#     return dcg


# def ndcg_at_k(rel, k):
#     """
#     calculate normalized discounted cumulative gain (ndcg)
#     rel: list, element is positive real values, can be binary
#     """
#     idcg = dcg_at_k(sorted(rel, reverse=True), k)
#     if not idcg:
#         return 0.
#     return dcg_at_k(rel, k) / idcg


# def ndcg_at_k_batch(hits, k):
#     """
#     calculate NDCG@k
#     hits: array, element is binary (0 / 1), 2-dim
#     """
#     hits_k = hits[:, :k]
#     dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

#     sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
#     idcg = np.sum((2 ** sorted_hits_k - 1) /
#                   np.log2(np.arange(2, k + 2)), axis=1)

#     idcg[idcg == 0] = np.inf
#     ndcg = (dcg / idcg)
#     return ndcg


# def recall_at_k(hit, k, all_pos_num):
#     """
#     calculate Recall@k
#     hit: list, element is binary (0 / 1)
#     """
#     hit = np.asfarray(hit)[:k]
#     return np.sum(hit) / all_pos_num


# def recall_at_k_batch(hits, k):
#     """
#     calculate Recall@k
#     hits: array, element is binary (0 / 1), 2-dim
#     """
#     res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
#     return res


# def F1(pre, rec):
#     if pre + rec > 0:
#         return (2.0 * pre * rec) / (pre + rec)
#     else:
#         return 0.


# def calc_auc(ground_truth, prediction):
#     try:
#         res = roc_auc_score(y_true=ground_truth, y_score=prediction)
#     except Exception:
#         res = 0.
#     return res


# def logloss(ground_truth, prediction):
#     logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
#     return logloss


# def calc_metrics_at_k(prediction_scores, train_head_dict, test_head_dict, head_ids, tail_ids, Ks):
#     """
#     prediction_scores: (n_heads, n_tails)
#     """
#     tail_ids = list(tail_ids)
#     test_pos_item_binary = np.zeros(
#         [len(head_ids), len(tail_ids)], dtype=np.float32)

#     for idx, h_id in enumerate(head_ids):
#         train_pos_tail_list = []
#         test_pos_tail_list = []

#         if h_id in train_head_dict:
#             train_pos_tail_list = train_head_dict[h_id]
#         if h_id in test_head_dict:
#             test_pos_tail_list = test_head_dict[h_id]

#         prediction_scores[idx][[tail_ids.index(x) for x in train_pos_tail_list]] = -np.inf
#         test_pos_item_binary[idx][[tail_ids.index(x) for x in test_pos_tail_list]] = 1

#     try:
#         # try to speed up the sorting process
#         _, rank_indices = torch.sort(prediction_scores.cuda(), descending=True)
#     except:
#         _, rank_indices = torch.sort(prediction_scores, descending=True)

#     rank_indices = rank_indices.cpu()

#     binary_hit = []
#     for i in range(len(head_ids)):
#         binary_hit.append(test_pos_item_binary[i][rank_indices[i]])

#     binary_hit = np.array(binary_hit, dtype=np.float32)

#     metrics_dict = {}
#     for k in Ks:
#         metrics_dict[k] = {}
#         metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
#         metrics_dict[k]['recall'] = recall_at_k_batch(binary_hit, k)
#     return metrics_dict

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
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    metrics_dict['accuracy'] = accuracy
    # precision tp / (tp + fp)
    precision = precision_score(y_pred=y_pred, y_true=y_true, average='weighted')
    metrics_dict['precision'] = precision
    # recall: tp / (tp + fn)
    recall = recall_score(y_pred=y_pred, y_true=y_true, average='weighted')
    metrics_dict['recall'] = recall
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
    metrics_dict['f1'] = f1
    return metrics_dict
