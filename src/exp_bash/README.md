## get_embedding.sh
```bash
ng=6 # 사용할 GPU 개수
bs=8 # Batch size
msl=2048 # 512 : 최대 token 길이
emb=eos # embedding 추출 방법 (masked : input mask, eos : 마지막 token embedding)
tn=task3_cat # embedding 추출 데이터 (task1_pro : amazon 상품, task1_cat : amazon 카테고리, task2_pro : 다나와 상품, task2_cat : 다나와 카테고리, task3_pro : 아마존, 다나와 상품, task3_cat : 조달청 물품 분류 체계)

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

eval $cmd
```

## eval_task12.sh
```bash
for lv in 4 3 2 1 # 맵핑할 카테고리 level (1~4 : 최상위 카테고리에서 최하위 카테고리)
do
for m in ft # TC-llama2 사용
do
for t in amazon # amazon : task1 수행, danawa : task2 수행
do
for msl in 2048 # 최대 token 길이
do
for emb in masked # eos (맵핑에 사용할 embedding 선택)
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
```

## eval_task3.sh
```bash
for c in 1 2 3 # case 선택
do
for e in eos masked # (맵핑에 사용할 embedding 선택)
do
for t in amazon danawa all # (임베딩 맵핑할 상품 선택. amazon : 아마존 50개 상품, danawa : 다나와 50개 상품, all : 전체 100개 상품)
do
for vl in 5 4 3 2 1 # 맵핑할 카테고리 level (1~5 : 최상위 카테고리에서 최하위 카테고리)
do
for m in ft # TC-llama2 사용
do
for l in 512 2048 # 최대 token 길이
do
for p in 1 # 2 : prompt version 1 또는 2
do
for nl in 50 25 40 # case 2 또는 3 에서, 필터링할 상위 카테고리 맵핑 개수 (예, 25개일 때, 상위 카테고리 임베딩들에서 상품 임베딩과 가까운 상위 25개 카테고리의 하위 카테고리들만 고려하여 임베딩 맵핑에 사용)
do
  case=$c
  emb=$e
  pn=$t
  lv=$vl
  msl=$l
  v=$p
  nhl=$nl

cmd="python train/eval_tasks.py \
--case_num $case \
--task zodal \
--pro_name $pn \
--level $lv \
--mode $m \
--max_source_length $msl \
--version $v"

if [ "$emb" = "masked" ]; then
  cmd="$cmd --input_mask"
elif [ "$emb" = "eos" ]; then
  cmd="$cmd --use_eos"
fi

if [ "$case" = "2" ] || [ "$case" = "3" ]; then
  cmd="$cmd --num_high_lv $nhl"
fi

eval $cmd

done
done
done
done
done
done
done
done
```