# Task 1,2 & 3

---

## TC-LLaMa 2 에서 product 또는 category discription 임베딩 추출

### 실행 명령어

```python
python src/train_bash.py --stage emb --model_name_or_path Llama-2-7b-chat-hf --do_predict --prompt_template llama2 --local_rank 0 --finetuning_type lora --checkpoint_dir checkpoints/checkpoint-38400 --per_device_eval_batch_size 8 --bf16 --predict_with_generate --lora_target q_proj,v_proj  --max_source_length 2048 --max_target_length 2048 --output_dir {output_save_path} --task {task_name} --dataset {dataset_name}
python src/train_bash.py --stage emb --model_name_or_path Llama-2-7b-chat-hf --do_predict --prompt_template llama2 --local_rank 0 --finetuning_type lora --checkpoint_dir checkpoints/checkpoint-38400 --per_device_eval_batch_size 8 --bf16 --predict_with_generate --lora_target q_proj,v_proj  --max_source_length 2048 --max_target_length 2048 --output_dir {output_save_path} --task {task_name} --dataset {dataset_name} --use_eos
python src/train_bash.py --stage emb --model_name_or_path Llama-2-7b-chat-hf --do_predict --prompt_template llama2 --local_rank 0 --finetuning_type lora --checkpoint_dir checkpoints/checkpoint-38400 --per_device_eval_batch_size 8 --bf16 --predict_with_generate --lora_target q_proj,v_proj  --max_source_length 2048 --max_target_length 2048 --output_dir {output_save_path} --task {task_name} --dataset {dataset_name} --input_mask
```

- --stage emb : embedding 추출을 위한 기본 세팅 설정
- model_name_or_path {model path} : LLama 2 7b chat 기본 모델 불러오기
- --checkpoint_dir {checkpiont path} : Fine-tuning 된 TC-LLama2 checkpoint 경로
- --max_source_length : Input token의 최대 길이, [512, 2048]
- --max_target_length : Output token의 최대 길이, [512, 2048]
- --output_dir {output embedding path} : 추출된 embedding 저장할 경로
- --task {task name} : 추출할 embedding의 task와 종류 (product 또는 category) 명
    - 아마존 상품 설명 : amazon_pro
    - 아마존 카테고리 설명 : amazon_cat
    - 다나와 상품 설명 : danawa_pro
    - 다나와 카테고리 설명 : danawa_cat
    - from 아마존/다나와 to 조달청 상품 설명 : zodal_pro
    - from 아마존/다나와 to 조달청 카테고리 설명 : zodal_cat
- --dataset {dataset name} : Load할 데이터 이름
    - Task1 (아마존 상품 설명 데이터) : kisti_task1
    - Task1 (아마존 카테고리 설명 데이터) : kisti_task1C
    - Task2 (다나와 상품 설명 데이터) : kisti_task2
    - Task2 (다나와 카테고리 설명 데이터) : kisti_task2C
    - Task3 (from 아마존/다나와 to 조달청 상품 설명) : kisti_task3
    - Task3 (from 아마존/다나와 to 조달청 카테고리 설명) : kisti_task3C
- --use_eos 또는 --input_mask : 임베딩 추출 위치 설정
    - use_eos : 가장 마지막 token embedding을 최종 임베딩으로 사용
    - input_mask : zero-padding token 제외 나머지 embedding을 평균 내어 최종 임베딩으로 사용
    - 설정 없을 시 : 모든 input embedding을 평균 내어 최종 임베딩으로 사용

라마 기본 코드와, train_bash, extract_emb.py

## Task 1, 2

```python
python src/task_12.py --task {task name} --k1 10 --k2 20 --k3 30 --level {level} --mode {fine-tuning} --max_source_length 512 --input_mask
```

- --task {task name} : Mapping 할 task 설정, [amazon, danawa]
- --k1 {top_k1} : Mapping 을 비교할 유사도 상위 카테고리 최소 개수
- --k2 {top_k2} : Mapping 을 비교할 유사도 상위 카테고리 중간 개수
- --k3 {top_k3} : Mapping 을 비교할 유사도 상위 카테고리 최대 개수
    - 본 실험에서는 10, 20, 30 으로 설정하여 수행.
- --level {level} : Mapping 을 수행할 카테고리 level 옵션 [1, 2, 3, 4]
- --mode {fine-tuning} : TC-Llama 2 추출 임베딩 사용시 [ft] , Zero-shot Llama 2 추출 임베딩 사용시 [zs]
- --max_source_length : 임베딩 추출시 최대 token 길이 옵션 선택 [512, 2048]
- --use_eos 또는 --input_mask : mapping에 사용할 임베딩 옵션 선택

task_12.py , utils_task_12.py

## Task 3

```python
python src/eval_tasks.py --case_num {case_num} --task zodal --pro_name {product_name} --level {level} --mode {fine-tuning} --max_source_length 512 --version {prompt_version} --input_mask
```

- --case_num {case_num} : 6가지 embedding mapping 방법들 중 하나 선택 [1,2,3,4,5,6]
- --task zodal : 명시적으로 Task 3 수행 설정
- --pro_name {product_name}
- --level {level} : Mapping 을 수행할 카테고리 level 옵션 [1, 2, 3, 4, 5]
- --mode {fine-tuning} :  TC-Llama 2 추출 임베딩 사용시 [ft] , Zero-shot Llama 2 추출 임베딩 사용시 [zs]
- --max_source_length : 임베딩 추출시 최대 token 길이 옵션 선택 [512, 2048]
- --version {prompt_version} : prompt version 선택 [1, 2]
- --use_eos 또는 --input_mask : mapping에 사용할 임베딩 옵션 선택