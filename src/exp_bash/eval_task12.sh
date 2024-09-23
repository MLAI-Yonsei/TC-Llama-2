#!/bin/bash
cd ..

for lv in 4 3 2 1
do
for m in ft
do
for t in amazon
do
for msl in 2048
do
for emb in pure # masked eos
do
  if [ "$emb" = "masked" ]; then
      CUDA_VISIBLE_DEVICES=4 python train/task_12.py --task $t --k1 10 --k2 20 --k3 30 --level $lv --mode $m --max_source_length $msl --input_mask
  elif [ "$emb" = "eos" ]; then
      CUDA_VISIBLE_DEVICES=4 python train/task_12.py --task $t --k1 10 --k2 20 --k3 30 --level $lv --mode $m --max_source_length $msl --use_eos
  else
      CUDA_VISIBLE_DEVICES=4 python train/task_12.py --task $t --k1 10 --k2 20 --k3 30 --level $lv --mode $m --max_source_length $msl
  fi
done
done
done
done
done