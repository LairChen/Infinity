from json import load
from os import system, listdir
from re import match

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from utils import *


def get_data_num() -> int:
    """动态获取训练数据集规模"""
    data = load(fp=open(file="data/train.json", encoding="utf-8"))
    return len(data)


def get_model_name() -> str:
    """动态获取基座模型名称"""
    for filename in listdir(path_train_pretrain):
        model = match(pattern="(.*)\.zip", string=filename)  # noqa
        if model is not None:
            return model.groups()[0]
    raise FileNotFoundError("No existing base model")


def merge_model_and_tokenizer() -> None:
    """合并模型和词表以构建HF标准模型"""
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=f"{path_train_pretrain}/tune",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_directory=path_train_finetune, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=f"{path_train_pretrain}/{get_model_name()}",
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_directory=path_train_finetune)
    return


cmd_mkdir = f"mkdir {path_train_pretrain}/tune"
cmd_train = "deepspeed --hostfile='' core/train.py " \
            "--report_to none " \
            f"--model_name_or_path {path_train_pretrain}/{get_model_name()} " \
            "--data_path data/train.json " \
            f"--output_dir {path_train_pretrain}/tune " \
            f"--model_max_length {llm['size']} " \
            f"--num_train_epochs {llm['epoch']} " \
            f"--per_device_train_batch_size {llm['batch']} " \
            "--save_strategy steps " \
            f"--save_steps {get_data_num() * llm['epoch'] // llm['batch']} " \
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
              f"--base {path_train_pretrain}/{get_model_name()} " \
              "--output /tmp/dataset/tune"

if __name__ == "__main__":
    system(cmd_mkdir)  # 创建模型微调的临时路径
    system(cmd_train)  # 模型微调
    system(cmd_predict)  # 模型预测
    merge_model_and_tokenizer()  # 合并微调参数和词表
