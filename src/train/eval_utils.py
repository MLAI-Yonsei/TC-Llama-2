import json
import numpy as np
import torch
import os

def reciprocal_rank_at_k(rank_list, relevant_items, k):
    for i, item in enumerate(rank_list):
        if (np.array(item) == np.array(relevant_items)).any():
            return 1.0 / (i + 1)
    return 0.0

def top_k_accuracy(y_pred_probs,y_true, k):
    y_true_list = [torch.tensor(y_true[idx]) for idx in sorted(y_true.keys())]

    # Stack the tensors to form a 1-dimensional tensor
    y_true_tensor = torch.stack(y_true_list)

    # Get the indices of the top-k predictions for each example
    top_k_pred_indices = y_pred_probs.cuda()

    # Convert true labels to a column vector to match the shape of top_k_pred_indices
    y_true_column = y_true_tensor.view(-1, 1).cuda()

    # Count the number of correct predictions where the true label is in the top-k predicted classes
    correct_predictions = torch.sum(torch.any(top_k_pred_indices == y_true_column, dim=1))

    # Calculate top-k accuracy
    top_k_acc = correct_predictions.item() / len(y_true_list)
    return top_k_acc*100

def recall_at_k(predictions, targets, k, task):
    # predictions: tensor of similarity scores (shape: num_products x num_categories)
    # targets: list or dictionary of relevant category indices for each product
    # k: top-k value for Recall@k

    top_k_indices = predictions
    num_correct = sum(any(target in top_k for target in targets[str(i)]) for i, top_k in enumerate(top_k_indices))
    total_relevant_items = sum(len(targets[str(i)]) for i in range(len(targets)))
    recall_at_k = num_correct / total_relevant_items

    correct = []
    for i, top_k in enumerate(top_k_indices):
        for target in targets[str(i)]:
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
        top_k = predictions[i].tolist()
        relevant_items = targets[str(i)]
        mrr_sum += reciprocal_rank_at_k(top_k, relevant_items, k)
    mrr_at_k = mrr_sum / num_queries
    return mrr_at_k*100

class Category_Label():
    def __init__(self, args):
        self.args = args

        # embedding path
        self.path = 'data/kisti_mapping/task3/'

        # product raw data
        if args.version==2:
            self.data_pro = json.load(open(os.path.join(self.path, f'task3_product_v2.json')))
        else:
            self.data_pro = json.load(open(os.path.join(self.path, f'task3_product.json')))

        # lv5 catevory raw data
        self.data_cat = json.load(open(os.path.join(self.path, f'zodal_cat.json')))

        self.level = args.level

        self.label_idx = json.load(open(os.path.join(self.path, f'lv{self.level}_label_idx.json')))

        if self.level < 5:
            self.lv5idx2high_lv_idx = json.load(open(os.path.join(self.path, f'lv5_idx2lv{self.level}_idx.json')))

    def get_higher_lv_pred(self, top_sim_indices, k):
        tmp = []
        for topk in top_sim_indices:
            ttt = {}
            for i in topk:
                high_lv_idx = self.lv5idx2high_lv_idx[str(i.item())]
                ttt[high_lv_idx]=0
                if len(ttt) == k:
                    ttt = list(ttt.keys())
                    break
            tmp.append(ttt)
        topk_level = torch.tensor(tmp)
        return topk_level

    def get_label(self):
        if self.args.pro_name == 'danawa':
            label_idx = {}
            for i in range(50):
                label_idx[str(i)]=self.label_idx[str(i)]
            return label_idx
        elif self.args.pro_name == 'amazon':
            label_idx = {}
            for k, i in enumerate(range(50, 100)):
                label_idx[str(k)] = self.label_idx[str(i)]
            return label_idx
        elif self.args.pro_name == 'all':
            return self.label_idx
        else:
            raise ValueError("Check the args.pro_name!!!")

class ReadytoEval():
    def __init__(self, args):
        self.args = args
        self.sim_score = None

        self.path_emb = f"kisti_output/"
        if args.mode == 'zs':
            self.path_emb += "zs/"

        self.set_eval(args.case_num)

    def get_pro_emb(self):
        pro_name = f"emb_task3_pro_max_leng_{str(self.args.max_source_length)}"

        if self.args.use_eos:
            pro_name += "_eos"

        elif self.args.input_mask:
            pro_name += "_masked"

        if self.args.version == 2:
            pro_name += "_v2"

        pro_embedding = torch.load(os.path.join(self.path_emb, f"{pro_name}.pt"), map_location='cpu')
        pro_emb_norm = pro_embedding / pro_embedding.norm(dim=-1, keepdim=True)

        if self.args.pro_name == 'amazon':
            pro_emb_norm = pro_emb_norm[50:]
            assert len(pro_emb_norm) == 50
        elif self.args.pro_name == 'danawa':
            pro_emb_norm = pro_emb_norm[:50]
            assert len(pro_emb_norm) == 50

        return pro_emb_norm

    def set_eval(self, case_num):
        if case_num == 1:
            # Standard way
            # Use embedding of level 5
            self.pro_emb_norm = self.get_pro_emb()
            cat_name = f"emb_zodal_cat_max_leng_{str(self.args.max_source_length)}"
            if self.args.use_eos:
                cat_name += "_eos"
            elif self.args.input_mask:
                cat_name += "_masked"

            cat_embedding = torch.load(os.path.join(self.path_emb,f"{cat_name}.pt"), map_location='cpu')
            self.cat_emb_norm = cat_embedding / cat_embedding.norm(dim=-1, keepdim=True)

        elif case_num == 2:
            # Filitering with higher level emb (kor UNSPSC).
            self.pro_emb_norm = self.get_pro_emb()
            cat_name = f"emb_zodal_cat_max_leng_{str(self.args.max_source_length)}"
            catlv1_name = f"emb_{self.args.task}_cat_lv1_max_leng_{str(self.args.max_source_length)}"

            if self.args.use_eos:
                cat_name += "_eos"
                catlv1_name += "_eos"
            elif self.args.input_mask:
                cat_name += "_masked"
                catlv1_name += "_masked"

            cat_embedding = torch.load(os.path.join(self.path_emb, f"{cat_name}.pt"), map_location='cpu')
            self.cat_emb_norm = cat_embedding / cat_embedding.norm(dim=-1, keepdim=True)

            catlv1_embedding = torch.load(os.path.join(self.path_emb, f"{catlv1_name}.pt"), map_location='cpu')
            self.catlv1_emb_norm = catlv1_embedding / catlv1_embedding.norm(dim=-1, keepdim=True)

            self.catlv1idx2lv1idx = json.load(open('data/kisti_mapping/task3/zodal_lv1idx2lv1idx.json'))
            self.lv1idx2lv5idx = json.load(open('data/kisti_mapping/task3/lv1idx2lv5idx.json'))

            self.high_lv_emb = self.catlv1_emb_norm
            self.highlv_idx2lv5_idx = self.lv1idx2lv5idx
            self.catlvidx2lvidx = self.catlv1idx2lv1idx

        elif case_num == 3:
            # Filitering with higher level emb (Eng. UNSPSC).
            self.pro_emb_norm = self.get_pro_emb()
            cat_name = f"emb_zodal_cat_max_leng_{str(self.args.max_source_length)}"
            unslv1_name = f"emb_unspsc_lv1_max_leng_{str(self.args.max_source_length)}"

            if self.args.use_eos:
                cat_name += "_eos"
                unslv1_name += "_eos"
            elif self.args.input_mask:
                cat_name += "_masked"
                unslv1_name += "_masked"

            cat_embedding = torch.load(os.path.join(self.path_emb, f"{cat_name}.pt"), map_location='cpu')
            self.cat_emb_norm = cat_embedding / cat_embedding.norm(dim=-1, keepdim=True)

            unslv1_embedding = torch.load(os.path.join(self.path_emb, f"{unslv1_name}.pt"), map_location='cpu')
            self.unslv1_emb_norm = unslv1_embedding / unslv1_embedding.norm(dim=-1, keepdim=True)

            self.unslv1idx2lv1idx = json.load(open('data/kisti_mapping/task3/unscp_lv1idx2lv1idx.json'))
            self.lv1idx2lv5idx = json.load(open('data/kisti_mapping/task3/lv1idx2lv5idx.json'))

            self.high_lv_emb = self.unslv1_emb_norm
            self.highlv_idx2lv5_idx = self.lv1idx2lv5idx
            self.unslvidx2lvidx = self.unslv1idx2lv1idx

    def compute_sim_score(self, case_num):
        if case_num == 1:
            self.pro_emb_norm = self.pro_emb_norm.cuda()
            self.cat_emb_norm = self.cat_emb_norm.cuda()
            self.sim_score = torch.matmul(self.pro_emb_norm, self.cat_emb_norm.T)
            return self.sim_score

        elif case_num==2:
            self.pro_emb_norm = self.pro_emb_norm.cuda()
            self.cat_emb_norm = self.cat_emb_norm.cuda()
            self.high_lv_emb = self.high_lv_emb.cuda()

            self.sim_emb = torch.matmul(self.cat_emb_norm, self.pro_emb_norm.T)

            high_lv_sim = torch.matmul(self.pro_emb_norm, self.high_lv_emb.T)
            _, top_k_indices = torch.topk(high_lv_sim, self.args.num_high_lv, dim=1)
            top_k_indices = top_k_indices.detach().cpu()

            self.filtered_lv5_candidate = {}
            min_cands_num = 10000
            for i, topk in enumerate(top_k_indices):
                ttt = []
                for k in topk:
                    n = self.catlvidx2lvidx[str(k.item())]
                    tmp = self.highlv_idx2lv5_idx[str(n)]
                    ttt.extend(tmp)

                if min_cands_num > len(ttt):
                    min_cands_num = len(ttt)
                self.filtered_lv5_candidate[i] = ttt

            self.sim_score = []
            for i in range(len(self.pro_emb_norm)):
                lv5_candidates = torch.tensor(self.filtered_lv5_candidate[i])
                tmp = self.sim_emb[lv5_candidates, i]
                _, top_indices = torch.topk(tmp, min_cands_num)
                top_indices = top_indices.cpu()
                top_indices = lv5_candidates[top_indices]
                self.sim_score.append(top_indices.tolist())

            # self.sim_score = torch.tensor(self.sim_score)
            # if self.sim_score.shape[-1] < 300:
            #     p = int(self.sim_score.shape[-1])
            # else:
            #     p = 500
            # _, top_sim_indices = torch.topk(self.sim_score, p, dim=1)

            top_sim_indices = torch.tensor(self.sim_score)
            return top_sim_indices

        elif case_num==3:
            self.pro_emb_norm = self.pro_emb_norm.cuda()
            self.cat_emb_norm = self.cat_emb_norm.cuda()
            self.high_lv_emb = self.high_lv_emb.cuda()

            self.sim_emb = torch.matmul(self.cat_emb_norm, self.pro_emb_norm.T)

            high_lv_sim = torch.matmul(self.pro_emb_norm, self.high_lv_emb.T)
            _, top_k_indices = torch.topk(high_lv_sim, self.args.num_high_lv, dim=1)
            top_k_indices = top_k_indices.detach().cpu()

            self.filtered_lv5_candidate = {}
            min_cands_num = 10000
            for i, topk in enumerate(top_k_indices):
                ttt = []
                for k in topk:
                    n = self.unslvidx2lvidx[str(k.item())]
                    tmp = self.highlv_idx2lv5_idx[str(n)]
                    ttt.extend(tmp)

                if min_cands_num > len(ttt):
                    min_cands_num = len(ttt)
                self.filtered_lv5_candidate[i] = ttt

            self.sim_score = []
            for i in range(len(self.pro_emb_norm)):
                lv5_candidates = torch.tensor(self.filtered_lv5_candidate[i])
                tmp = self.sim_emb[lv5_candidates, i]
                _, top_indices = torch.topk(tmp, min_cands_num)
                top_indices = top_indices.cpu()
                top_indices = lv5_candidates[top_indices]
                self.sim_score.append(top_indices.tolist())

            # self.sim_score = torch.tensor(self.sim_score)
            # if self.sim_score.shape[-1] < 300:
            #     p = int(self.sim_score.shape[-1])
            # else:
            #     p = 500
            # _, top_sim_indices = torch.topk(self.sim_score, p, dim=1)

            top_sim_indices = torch.tensor(self.sim_score)
            return top_sim_indices