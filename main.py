from argparse import ArgumentParser
from os import system

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

data_path = "/tmp/dataset"
tune_path = "/tmp/output"
parser = ArgumentParser()
parser.add_argument("--model", "-m", help="base model name")
args = parser.parse_args()
model_name = args.model

cmd_mkdir = "mkdir /tmp/dataset/tune"
cmd_unzip = f"unzip {data_path}/{model_name}.zip -d {data_path}"  # noqa
cmd_train = "deepspeed --hostfile='' core/train.py " \
            "--report_to none " \
            f"--model_name_or_path {data_path}/{model_name} " \
            "--data_path data/train.json " \
            "--output_dir /tmp/dataset/tune " \
            "--model_max_length 1024 " \
            "--num_train_epochs 16 " \
            "--per_device_train_batch_size 1 " \
            "--save_strategy steps " \
            "--save_steps 976 " \
            "--save_total_limit 1 " \
            "--lr_scheduler_type constant " \
            "--learning_rate 2e-5 " \
            "--adam_beta1 0.9 " \
            "--adam_beta2 0.98 " \
            "--adam_epsilon 1e-8 " \
            "--max_grad_norm 1.0 " \
            "--warmup_ratio 0.0 " \
            "--weight_decay 1e-4 " \
            "--logging_steps 1 " \
            "--gradient_accumulation_steps 1 " \
            "--gradient_checkpointing True " \
            "--deepspeed data/deepspeed.json " \
            "--bf16 True " \
            "--tf32 False " \
            "--use_lora True"
cmd_predict = "python core/predict.py " \
              f"--base {data_path}/{model_name} " \
              f"--output /tmp/dataset/tune"


def merge_model_and_tokenizer() -> None:
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="/tmp/dataset/tune",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_directory="tune", safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/tmp/dataset/Baichuan2-13B-Chat",
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_directory="/tmp/output")
    return


if __name__ == "__main__":
    system(cmd_mkdir)
    system(cmd_unzip)  # 预训练模型解压缩
    system(cmd_train)  # 模型微调
    system(cmd_predict)  # 模型预测
    merge_model_and_tokenizer()
