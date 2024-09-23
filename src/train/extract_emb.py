import os
import torch
import json
import numpy as np
from tqdm import tqdm
from typing import Optional, List
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from transformers.trainer_pt_utils import nested_concat

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneralArguments
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from llmtuner.tuner.sft.trainer import Seq2SeqPeftTrainer

IGNORE_INDEX = -100

def run_emb(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    general_args: GeneralArguments,
    callbacks: Optional[List[TrainerCallback]] = [LogCallback()]
):
    dataset = get_dataset(model_args, data_args)

    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.eval_num_beams if \
                data_args.eval_num_beams is not None else training_args.generation_num_beams

    # Initialize our Trainer
    trainer = Seq2SeqPeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args)
    )

    dataloader = trainer.get_test_dataloader(dataset)

    print("Length of data : ", len(dataloader))
    total = len(dataloader)

    model.eval()
    avg_emb = None
    
    # 임베딩 저장 경로 설정
    max_input_lenght = -1
    for i, s in tqdm(enumerate(dataloader), total=total):
        inputs = trainer._prepare_input(s)
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to(model.device)

        if max_input_lenght < input_ids.shape[-1]:
            max_input_lenght = input_ids.shape[-1]

        if general_args.input_mask:
            input_mask = inputs.attention_mask.detach().cpu()
            input_length = input_mask.sum(dim=1)
            b, max_leng = input_mask.shape

        with torch.no_grad():
            embeddings = model.model(input_ids)[0]

            if general_args.input_mask:
                embeddings = embeddings.detach().cpu()
                emb_masked = embeddings * input_mask.view(b, max_leng, -1) # batch 내에서 입력 문장별 zero-padding 부분 마스킹
                emb_masked = emb_masked.sum(dim=1) # 마스킹 된 token 은 제외하고 나머지 각 token 별 embedding 합
                mean = emb_masked / input_length.view(b, -1) # 각 샘플별 입력  길이 중 making 부분 제외 유의미한 단어 개수로 나누어 평균 embedding 계산
            elif general_args.use_eos:
                mean = embeddings[:, -1, :] # 각 샘플별 마지막 token의 embedding만 추출
            else:
                mean = torch.mean(embeddings, 1).cpu().detach() # 각 샘플별 전체 문장의 embedding 평균

            avg_emb = mean if avg_emb is None else nested_concat(avg_emb, mean)

    if general_args.old_checkpoint:
        pt_name = f"old_emb_{general_args.task}_max_leng_{str(data_args.max_source_length)}"
    else:
        pt_name = f"emb_{general_args.task}_max_leng_{str(data_args.max_source_length)}"

    if general_args.input_mask:
        pt_name += "_masked"

    if general_args.use_eos:
        pt_name += "_eos"

    print("general_args.task : ", general_args.task)
    print("Max input token length : ", max_input_lenght)

    # 추출된 임베딩 저장.
    if model_args.checkpoint_dir is None:
        print(f"{general_args.task}'s Embedding size : {avg_emb.shape}")
        torch.save(avg_emb, f"./kisti_output/zs/{pt_name}.pt")
    else:
        print(f"{general_args.task}'s Embedding size : {avg_emb.shape}")
        torch.save(avg_emb, f"./kisti_output/{pt_name}.pt")