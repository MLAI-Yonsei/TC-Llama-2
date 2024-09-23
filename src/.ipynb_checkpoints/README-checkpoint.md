# LLaMA Efficient Tuning
참고 REPO https://github.com/hiyouga/LLaMA-Efficient-Tuning

# Huggingface 모델을 가져오는 weigth 신청
 - 신청 주소 : https://ai.meta.com/resources/models-and-libraries/llama-downloads/

# 최종 학습된 Weight
  - checkpoint-38400

## Provided Datasets
- KISTI Fine-tuning:
  - data/kisti_train.json

# 실험을 위한 가상환경 설치
```bash
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
cd LLaMA-Efficient-Tuning
conda create -n llama_etuning python=3.10
conda activate llama_etuning
pip install -r requirements.txt
```

## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- sentencepiece and tiktoken
- jieba, rouge-chinese and nltk (used at evaluation)
- gradio and matplotlib (used in web_demo.py)
- uvicorn, fastapi and sse-starlette (used in api_demo.py)

## 학습전 dataset 준비

- 학습을 위한 KISTI 데이터 옮기기

  - /preparing_dataset/fine_tuning_data 에서 kisti_train.json 과 kisti_test.json 을 /src/data/ 로 복사

- Validation data 생성하기.

  - 학습 데이터 (kisti_train.json) 를 task 마다 동일한 비율로 train 과 validation data로 구분. 

    ```bash
    python data_split.py
    ```

## 사용할 Dataset 등록

- LLaMA-Efficient-Tuning/data 에 data 옮기기
- LLaMA-Efficient-Tuning/data/data_info.json 파일에 아래처럼 데이터 등록

```bash
  "dataset_name": {
    "file_name": "data_file_name.json" },
 *shard는 지정 안 해도됨
```

## 여러개의 GPU를 이용하여 학습
- Deepspeed 이용
- 여러개의 gpu를 사용할 경우 deepspeed를 이용할수 있다.
- 사용한 deepspeed config 파일 해당 config는 ds_deepspeed.json에 저장

```bash
  pip install deepspeed
```


```json
  {
      "train_micro_batch_size_per_gpu": "auto",
      "gradient_accumulation_steps": "auto",
      "gradient_clipping": "auto",
      "zero_allow_untested_optimizer": true,
      "fp16": {
        "enabled": "auto",
        "loss_scale": 1,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
      },  
      "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": false,
        "contiguous_gradients": true
      }
    }
```

### 학습 커맨드
```bash
deepspeed –-num_gpus 8 

train/train_bash.py
   --local_rank=0 \
   --deepspeed ds_deepspeed.json \
   --stage sft \ #supervised tuning
   --model_name_or_path Llama-2-7b-chat-hf \ # huggingface 모델 weight 경로 
   --do_train \ #  학습을 한다는 command
   --dataset kisti_data \ # 학습할 데이터셋 지정
   --template llama2 \ # template을 지정할수 있음, llama2를 사용하므로 llama2의 template 사용
   --finetuning_type lora \ # 빠른 학습을 위한 lora 방식 사용
   --lora_target q_proj,v_proj \
   --output_dir /path/to/save \ # 최종 모델을 저장할 경로를 지정
   --per_device_train_batch_size 2 \ # train dataset의 batchsize 지정(입력길이를 길게 설정했기때문에 batch 크기가 작음)
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --logging_steps 1000 \
   --save_steps 1920 \
   --learning_rate 1e-6 \
   --plot_loss \
   --bf16 \ # 빠른 학습을 위한 precision 설정
   --val_size 0.1 \
   --do_eval \ # 검증 단계가 필요할시 추가
   --evaluation_strategy steps \
   --eval_steps 2000 \
   --num_train_epochs 10 \ #학습 에폭 
   --max_source_length 2048 \ # 입력길이
   --max_target_length 2048 \ # 출력길이
   --overwrite_cache \ # cache overwrtie 유무
   --checkpoint_dir /path/checkpoints # tuning할 모델의 weight
```

### 생성 커맨드

```bash
CUDA_VISIBLE_DEVICES=0 
    python train/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_predict \
    --dataset kisti_test \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --predict_with_generate
```
## Embedding mapping (상품-카테고리 맵핑)
- train/extract_emb.py : 상품, 카테고리 임베딩 추출 코드
- train/task_12.py : task 1,2 (Amazon 상품 - 아마존 카테고리 맵핑, 다나와 상품 - 다나와 카테고리 맵핑)
- train/eval_tasks.py : task 3 (Amazon, 단나와 50개 상품 - 전자상거래 표준분류체계 맵핑)

### 임베딩 생성
- exp_bash/get_embdding.sh

### Task 1, 2, 3 수행
- exp_bash/eval_task12.sh
- exp_bash/eval_task3.sh

### Task 4 수행
생성 커맨드와 동일하나 task generating폴더의 실행파일에 의해 생성된 taskXX.json파일을 input값으로 하여 생성
프롬프트 변경을 원할 경우, task4_generate.ipynb에서 프롬프트를 교체하면됨.
```bash
CUDA_VISIBLE_DEVICES=0
    python train/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_predict \
    --dataset kistitaskXX \ #kistitaskXX.json task4 데이터를 이용하여 생성을 진행 #데이터셋 등록해야함
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --predict_with_generate
```