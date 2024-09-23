import os
import torch
import wandb
import argparse
from eval_utils import Category_Label, ReadytoEval, recall_at_k, mean_reciprocal_rank_at_k

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='zodal')
parser.add_argument("--pro_name", type=str, default='all', help="danawa, amazon, or all")
parser.add_argument("--version", type=int, default=1, help="v2 : add sentence to the last of product")
parser.add_argument("--k1", type=int, default=10)
parser.add_argument("--k2", type=int, default=20)
parser.add_argument("--k3", type=int, default=30)
parser.add_argument('--mode', type=str, default='zs')
parser.add_argument('--level', type=int, default=4, help='level 1, 2, 3, 4')
parser.add_argument('--max_source_length', type=int, default=512, help='512 or 2048')
parser.add_argument('--input_mask', action='store_true')
parser.add_argument('--use_eos', action='store_true')
parser.add_argument('--num_high_lv', type=int, default=1)
parser.add_argument('--case_num', type=int, default=0, help="[1,2,3]")

args = parser.parse_args()
k1 = args.k1
k2 = args.k2
k3 = args.k3

r2e = ReadytoEval(args)
catlabel = Category_Label(args)

wandb.init(
    project="TC-LLama2 Eval Task_KISTI2023_v5",
)
wandb.config.update(args)

ground_truth = catlabel.get_label()

if args.case_num==1 and args.level == 5:
    sim_score = r2e.compute_sim_score(args.case_num)
    _, topk1_cand = torch.topk(sim_score, k1, dim=1)
    _, topk2_cand = torch.topk(sim_score, k2, dim=1)
    _, topk3_cand = torch.topk(sim_score, k3, dim=1)

elif args.case_num==1 and args.level < 5:
    sim_score = r2e.compute_sim_score(args.case_num)

    p = sim_score.shape[-1]
    _, top_sim_indices = torch.topk(sim_score, p, dim=1)

    topk1_cand = catlabel.get_higher_lv_pred(top_sim_indices, k1)
    topk2_cand = catlabel.get_higher_lv_pred(top_sim_indices, k2)
    topk3_cand = catlabel.get_higher_lv_pred(top_sim_indices, k3)

elif args.case_num in [2,3] and args.level == 5:
    top_sim_indices = r2e.compute_sim_score(args.case_num)
    topk1_cand = top_sim_indices[:, :k1]
    topk2_cand = top_sim_indices[:, :k2]
    topk3_cand = top_sim_indices[:, :k3]

elif args.case_num in [2,3] and args.level < 5:
    top_sim_indices = r2e.compute_sim_score(args.case_num)
    topk1_cand = catlabel.get_higher_lv_pred(top_sim_indices, k1)
    topk2_cand = catlabel.get_higher_lv_pred(top_sim_indices, k2)
    topk3_cand = catlabel.get_higher_lv_pred(top_sim_indices, k3)

recall_at_k1_score = recall_at_k(topk1_cand, ground_truth, k1, args.task)
recall_at_k2_score = recall_at_k(topk2_cand, ground_truth, k2, args.task)
recall_at_k3_score = recall_at_k(topk3_cand, ground_truth, k3, args.task)

mrr_score_k1 = mean_reciprocal_rank_at_k(topk1_cand, ground_truth, k1, args.task)
mrr_score_k2 = mean_reciprocal_rank_at_k(topk2_cand, ground_truth, k2, args.task)
mrr_score_k3 = mean_reciprocal_rank_at_k(topk3_cand, ground_truth, k3, args.task)

import sys
sys.stdout = open(f'./kisti_output/results/case_num_{args.case_num}_{args.mode}_{args.pro_name}_lv{args.level}_v{args.version}.txt', 'w')

print('-------------------------------')
print(f'R@{k1}(%):', recall_at_k1_score)
print(f'R@{k2}(%):', recall_at_k2_score)
print(f'R@{k3}(%):', recall_at_k3_score)
print('-------------------------------')
print(f'MRR@{k1}(%):', mrr_score_k1)
print(f'MRR@{k2}(%):', mrr_score_k2)
print(f'MRR@{k3}(%):', mrr_score_k3)
print('-------------------------------')

wandb.log(
    {
       f'R@{k1}(%)':recall_at_k1_score,
       f'R@{k2}(%)':recall_at_k2_score,
       f'R@{k3}(%)':recall_at_k3_score,
       f'MRR@{k1}(%)':mrr_score_k1,
       f'MRR@{k2}(%)':mrr_score_k2,
       f'MRR@{k3}(%)':mrr_score_k3,
    }
)
sys.stdout.close()