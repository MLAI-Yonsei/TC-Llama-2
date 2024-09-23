#!/bin/bash
cd ..

ng=6
bs=8
msl=2048

tn=task3_cat

# Determine the appropriate values based on the condition
if [ "$tn" = "task1_pro" ]; then
  t=amazon_pro
  ds=kisti_task1
  outdir=./outputs/task1_pro
elif [ "$tn" = "task2_pro" ]; then
  t=danawa_pro
  ds=kisti_task2
  outdir=./outputs/danawa_pro
elif [ "$tn" = "task3_pro" ]; then
  t=task3_pro
  ds=kisti_task3
  outdir=./outputs/task3_pro
elif [ "$tn" = "task1_cat" ]; then
  t=amazon_cat
  ds=kisti_task1C
  outdir=./outputs/amazon_cat
elif [ "$tn" = "task2_cat" ]; then
  t=danawa_cat
  ds=kisti_task2C
  outdir=./outputs/danawa_cat
elif [ "$tn" = "task3_cat" ]; then
  t=zodal_cat
  ds=kisti_task3C
  outdir=./outputs/zodal_cat
fi

emb=eos

# Build the command with conditional arguments
cmd="deepspeed --num_gpus $ng train/train_bash.py --stage emb \
  --deepspeed ds_deepspeed.json \
  --model_name_or_path Llama-2-7b-chat-hf \
  --do_predict \
  --template llama2 \
  --local_rank 0 \
  --finetuning_type lora \
  --checkpoint_dir checkpoints/checkpoint-38400 \
  --per_device_eval_batch_size $bs \
  --bf16 \
  --predict_with_generate \
  --lora_target q_proj,v_proj \
  --max_source_length $msl \
  --output_dir $outdir \
  --task $t \
  --dataset $ds"

# Add conditional arguments based on the value of $emb
if [ "$emb" = "masked" ]; then
  cmd="$cmd --input_mask"
elif [ "$emb" = "eos" ]; then
  cmd="$cmd --use_eos"
fi

# Execute the built command
eval $cmd