from llmtuner import run_exp
from llmtuner.tuner import get_train_args
from extract_emb import run_emb

def main():
    model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args()
    if general_args.stage=='emb':
        # Embedding 추출
        pt_name = f"emb_{general_args.task}_max_leng_{str(data_args.max_source_length)}"
        
        if general_args.input_mask:
            pt_name += "_masked"

        if general_args.use_eos:
            pt_name += "_eos"

        if model_args.checkpoint_dir is None: # checkpoint_dir이 없을시 zero-shot Llama 2-7b-chat으로 임베딩 추출
            output_path = f"./kisti_output/zs/{pt_name}_v2.pt"
        else:
            output_path = f"./kisti_output/{pt_name}_v2.pt"

        import os
        if os.path.exists(output_path): # 이미 추출된 임베딩은 중복 추출하지 않음.
            print(f"{pt_name}_v2 is already exist")
            return None
        else:
            run_emb(model_args=model_args, data_args=data_args, training_args=training_args, finetuning_args=finetuning_args, general_args=general_args)

        run_emb(model_args=model_args, data_args=data_args, training_args=training_args, finetuning_args=finetuning_args, general_args=general_args)
    else:
        run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
