#  Tc-llama 2: fine-tuning LLM for technology and commercialization applications

## Abstract
This paper introduces TC-Llama 2, a novel application of large language models (LLMs) in the technology-commercialization field. Traditional methods in this field, reliant on statistical learning and expert knowledge, often face challenges in process ing the complex and diverse nature of technology-commercialization data. TC-Llama 2 addresses these limitations by utilizing the advanced generalization capabili ties of LLMs, specifically adapting them to this intricate domain. Our model, based on the open-source LLM framework, Llama 2, is customized through instruction tuning using bilingual Korean-English datasets. Our approach involves transforming technol ogy-commercialization data into formats compatible with LLMs, enabling the model to learn detailed technological knowledge and product hierarchies effectively. We introduce a unique model evaluation strategy, leveraging new matching and genera tion tasks to verify the alignment of the technology-commercialization relationship in TC-Llama 2. Our results, derived from refining task-specific instructions for inference, provide valuable insights into customizing language models for specific sectors, poten tially leading to new applications in technology categorization, utilization, and predic tive product development

## Overview
The preparing_dataset folder has data preprocessing and data generation code.
The src folder has Llama architecture code for fine-tuning and inference and mapping category.
Each folder has more specific instructions.


## Setup
```bash
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
cd LLaMA-Efficient-Tuning
conda create -n tc_llama python=3.10
conda activate tc_llama
pip install -r requirements.txt
pip install deepspeed
```

## Data Preparation
```bash
# Copy training data
cp /path/to/kisti_train.json /path/to/kisti_test.json src/data/

# Split data into train and validation sets
python data_split.py
```

## Training
```bash
deepspeed --num_gpus 8 train/train_bash.py \
    --stage sft \
    --model_name_or_path Llama-2-7b-chat-hf \
    --do_train \
    --dataset kisti_data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir /path/to/save \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --num_train_epochs 10 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --deepspeed ds_deepspeed.json
```

## Generation
```bash
CUDA_VISIBLE_DEVICES=0 python train/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_predict \
    --dataset kisti_test \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --predict_with_generate
```

## Embedding Mapping
```bash
# Extract embeddings
python train/extract_emb.py

# Perform mapping tasks
python train/task_12.py
python train/eval_tasks.py
```

## Task-Specific Data Generation
```bash
# For Task 1 & 2
python task12_generating/task1_amazon_product_prompt.py
python task12_generating/task1_amazon_category_prompt.py
python task12_generating/task2_danawa_product_prompt.py
python task12_generating/task2_danawa_category_prompt.py

# For Task 3
python task3_generating/amazon50_mapping_prompt.py
python task3_generating/danawa50_mapping_prompt.py
python task3_generating/zodalcheong_mapping_prompt.py

# For Task 4
python task4_generating/task4_generate.py
```

## Evaluation
```bash
# For Task 4
CUDA_VISIBLE_DEVICES=0 python train/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_predict \
    --dataset kistitaskXX \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --predict_with_generate
```

Note: Replace placeholders (e.g., `/path/to/...`) with actual paths in your environment. Adjust parameters as needed for your specific setup and requirements.