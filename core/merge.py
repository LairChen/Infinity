from argparse import ArgumentParser

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def merge_model_and_tokenizer(inputPath: str, outputPath: str, modelName: str) -> None:
    """合并模型和词表以构建HF标准模型"""
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="{}/tune".format(inputPath),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval().merge_and_unload()
    model.save_pretrained(save_directory=outputPath, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="{}/{}".format(inputPath, modelName),
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_directory=outputPath)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", help="temporary path for fine-tuned model")
    parser.add_argument("--output", "-o", help="eventual path for fine-tuned model")
    parser.add_argument("--model", "-m", help="pretrained model name")  # noqa
    args = parser.parse_args()
    merge_model_and_tokenizer(inputPath=args.input, outputPath=args.output, modelName=args.model)
