import os
import wandb
import argparse
from utils_task_12 import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='amazon')
parser.add_argument("--k1", type=int, default=10)
parser.add_argument("--k2", type=int, default=20)
parser.add_argument("--k3", type=int, default=30)
parser.add_argument('--mode', type=str, default='ft')
parser.add_argument('--level', type=int, default=4, help='level 1, 2, 3, 4')
parser.add_argument('--max_source_length', type=int, default=512, help='512 or 2048')
parser.add_argument('--input_mask', action='store_true')
parser.add_argument('--use_eos', action='store_true')

args = parser.parse_args()

k1 = args.k1
k2 = args.k2
k3 = args.k3

task = args.task

catlabel = Category_Label(task, lv=args.level)

wandb.init(
    project="TC-LLama2 Eval Task_KISTI2023_Task12_v5",
)
wandb.config.update(args)

pro_name = f"emb_{args.task}_pro_max_leng_{str(args.max_source_length)}"
cat_name = f"emb_{args.task}_cat_max_leng_{str(args.max_source_length)}"

if args.input_mask:
    pro_name += "_masked"
    cat_name += "_masked"

if args.use_eos:
    pro_name += "_eos"
    cat_name += "_eos"

if args.mode == 'ft':
    product_embedding_path = f'kisti_output/{pro_name}.pt'
    cat_embedding_path = f'kisti_output/{cat_name}.pt'
elif args.mode == 'zs':
    product_embedding_path = f'kisti_output/zs/{pro_name}.pt'
    cat_embedding_path = f'kisti_output/zs/{cat_name}.pt'

# torch.cuda.device
pro_embeddings = torch.load(product_embedding_path, map_location='cpu')
cat_embeddings = torch.load(cat_embedding_path, map_location='cpu')

pro_emb = pro_embeddings/pro_embeddings.norm(dim=-1, keepdim=True)
cat_emb = cat_embeddings/cat_embeddings.norm(dim=-1, keepdim=True)

pro_emb = pro_emb.cuda()
cat_emb = cat_emb.cuda()

if k1 > len(cat_emb):
    k1 = len(cat_emb)
    print(f"{k1} over the max length")
if k2 > len(cat_emb):
    k2 = len(cat_emb)
    print(f"{k2} over the max length")
if k3 > len(cat_emb):
    k3 = len(cat_emb)
    print(f"{k3} over the max length")

if args.level < 4:
    ground_truth = catlabel.get_label()
    ground_truth = catlabel.get_higher_cat(ground_truth)
else:
    ground_truth = catlabel.get_label()  # Replace this with your actual ground truth

if args.level < 4:
    sim_score = torch.matmul(pro_emb, cat_emb.T)

    topk1_cand = catlabel.get_higher_lv_pred(sim_score, k1)
    topk2_cand = catlabel.get_higher_lv_pred(sim_score, k2)
    topk3_cand = catlabel.get_higher_lv_pred(sim_score, k3)

elif args.level >= 4:
    sim_score = torch.matmul(pro_emb, cat_emb.T)
    _, topk1_cand = torch.topk(sim_score, k1, dim=1)
    _, topk2_cand = torch.topk(sim_score, k2, dim=1)
    _, topk3_cand = torch.topk(sim_score, k3, dim=1)

recall_at_k1_score = recall_at_k(topk1_cand, ground_truth, k1, task)
recall_at_k2_score = recall_at_k(topk2_cand, ground_truth, k2, task)
recall_at_k3_score = recall_at_k(topk3_cand, ground_truth, k3, task)

mrr_score_k1 = mean_reciprocal_rank_at_k(topk1_cand, ground_truth, k1, task)
mrr_score_k2 = mean_reciprocal_rank_at_k(topk2_cand, ground_truth, k2, task)
mrr_score_k3 = mean_reciprocal_rank_at_k(topk3_cand, ground_truth, k3, task)

import sys
sys.stdout = open(f'./kisti_output/results/{args.mode}_{pro_name}_lv{args.level}.txt', 'w')

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