#!/bin/bash

cd ..

for c in 1 2 3
do
for e in eos masked
do
for t in amazon danawa all
do
for vl in 5 4 3 2 1
do
for m in ft
do
for l in 512 2048
do
for p in 1
do
for nl in 50 25 40
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

# Add conditional arguments based on the value of $emb
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