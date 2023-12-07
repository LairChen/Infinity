from json import load
from os import system, listdir, mkdir
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
    model = model.merge_and_unload()
    model.save_pretrained(save_directory=path_train_finetune, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=f"{path_train_pretrain}/{get_model_name()}",
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_directory=path_train_finetune)
    del model, tokenizer
    return


cmd_train = "deepspeed core/train.py " \
            "--report_to none " \
            "--data_path data/train.json " \
            f"--model_name_or_path {path_train_pretrain}/{get_model_name()} " \
            f"--output_dir {path_train_pretrain}/tune " \
            "--save_total_limit 1 " \
            "--save_strategy steps " \
            f"--save_steps {get_data_num() * llm['num_train_epochs'] // llm['per_device_train_batch_size']} " \
            f"--model_max_length {llm['model_max_length']} " \
            f"--num_train_epochs {llm['num_train_epochs']} " \
            f"--per_device_train_batch_size {llm['per_device_train_batch_size']} " \
            f"--lr_scheduler_type {llm['lr_scheduler_type']} " \
            f"--learning_rate {llm['learning_rate']} " \
            f"--adam_beta1 {llm['adam_beta1']} " \
            f"--adam_beta2 {llm['adam_beta2']} " \
            f"--adam_epsilon {llm['adam_epsilon']} " \
            f"--max_grad_norm {llm['max_grad_norm']} " \
            f"--warmup_ratio {llm['warmup_ratio']} " \
            f"--weight_decay {llm['weight_decay']} " \
            "--deepspeed data/deepspeed.json " \
            "--logging_steps 1 " \
            "--gradient_accumulation_steps 1 " \
            "--gradient_checkpointing True " \
            "--bf16 True " \
            "--tf32 False " \
            "--use_lora True"
cmd_predict = "python core/predict.py " \
              f"--model {path_train_finetune}"

if __name__ == "__main__":
    mkdir(f"{path_train_pretrain}/tune")  # 创建模型微调的临时路径
    system(cmd_train)  # 模型微调
    merge_model_and_tokenizer()  # 合并微调参数和词表
    system(cmd_predict)  # 模型预测
