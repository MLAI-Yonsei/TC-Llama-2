import json
import pandas as pd
import numpy as np
import torch

class Category_Label():
    def __init__(self, task, lv):
        if task == 'amazon':
            n = 1
            self.task_ = task
        elif task == 'danawa':
            n = 2
            self.task_ = task

        self.path = f'data/kisti_mapping/task{n}/{task}_cat.json'
        self.data = json.load(open(self.path, encoding='utf-8'))

        self.path_p = f'data/kisti_mapping/task{n}/{self.task_}_product.json'
        self.data_p = json.load(open(self.path_p, encoding='utf-8'))

        if lv < 4:
            self.path_l = f'data/kisti_mapping/task{n}/'
            self.idx2label = json.load(open(self.path_l + 'idx2label.json', encoding='utf-8'))
            self.label2level = json.load(open(self.path_l+f'label2level{lv}.json', encoding='utf-8'))
            self.lv2idx = json.load(open(self.path_l+f'lv{lv}toidx.json', encoding='utf-8'))

        self.level = lv

    def get_higher_cat(self, gt):
        # gt : dict (idx : [label])
        # (100, 1)
        level_gt = {}
        for k, v in gt.items():
            # label = self.idx2label[str(v[0])]
            level = self.label2level[str(v[0])]
            idx = self.lv2idx[level]
            level_gt[k]=[idx]
        return level_gt

    def get_higher_lv_pred(self, sim_score, k):
        _, top_k_indices = torch.topk(sim_score, k, dim=1)
        tmp = []
        for topk in top_k_indices:
            ttt = []
            for k in topk:
                level = self.label2level[str(k.item())]
                ttt.append(self.lv2idx[level])
            tmp.append(ttt)

        topk_level = torch.tensor(tmp)
        return topk_level

    def get_label(self):
        out_cat = []
        for i in range(len(self.data)):
            out_cat.append(self.data[i]["output"].replace('products', 'product'))

        self.df = pd.DataFrame({"out_cat": out_cat})

        out_pro = []
        for i in range(len(self.data_p)):
            out_pro.append(self.data_p[i]["output"].replace('products', 'product'))
        # df_p = pd.DataFrame({"out_cat": out_pro})

        label = {}
        for i, sen in enumerate(out_pro):
            label[i] = [self.df[self.df['out_cat'] == sen].index[0]]

        return label

def top_k_accuracy(y_pred_probs,y_true, k):
    y_true_list = [torch.tensor(y_true[idx]) for idx in sorted(y_true.keys())]

    # Stack the tensors to form a 1-dimensional tensor
    y_true_tensor = torch.stack(y_true_list)

    # Get the indices of the top-k predictions for each example
    # top_k_pred_indices = torch.argsort(y_pred_probs, dim=1)[:, -k:]
    top_k_pred_indices = y_pred_probs.cuda()

    # Convert true labels to a column vector to match the shape of top_k_pred_indices
    y_true_column = y_true_tensor.view(-1, 1).cuda()

    # Count the number of correct predictions where the true label is in the top-k predicted classes
    correct_predictions = torch.sum(torch.any(top_k_pred_indices == y_true_column, dim=1))

    # Calculate top-k accuracy
    top_k_acc = correct_predictions.item() / len(y_true_list)
    return top_k_acc*100

def reciprocal_rank_at_k(rank_list, relevant_items, k):
    for i, item in enumerate(rank_list):
        if (np.array(item) == np.array(relevant_items)).any():
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(predictions, targets, k, task):
    # predictions: tensor of similarity scores (shape: num_products x num_categories)
    # targets: list or dictionary of relevant category indices for each product
    # k: top-k value for Recall@k

    # _, top_k_indices = torch.topk(predictions, k, dim=1)
    top_k_indices = predictions
    num_correct = sum(any(target in top_k for target in targets[i]) for i, top_k in enumerate(top_k_indices))
    total_relevant_items = sum(len(targets[i]) for i in range(len(targets)))
    recall_at_k = num_correct / total_relevant_items

    correct = []
    for i, top_k in enumerate(top_k_indices):
        for target in targets[i]:
            if target in top_k:
                correct.append(i)

    print(f'recall@{k}', correct)
    return recall_at_k * 100

def mean_reciprocal_rank_at_k(predictions, targets, k, task):
    # predictions: tensor of similarity scores (shape: num_queries x num_items)
    # targets: list or dictionary of relevant item indices for each query1
    mrr_sum = 0.0
    num_queries = len(targets)
    for i in range(num_queries):
        # top_k = torch.topk(predictions[i], k)[1].tolist()
        top_k = predictions.tolist()
        #rank_list = torch.argsort(predictions[i], descending=True).tolist()
        relevant_items = targets[i]
        mrr_sum += reciprocal_rank_at_k(top_k, relevant_items, k)
    mrr_at_k = mrr_sum / num_queries
    return mrr_at_k*100