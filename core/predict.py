from argparse import ArgumentParser
from json import load
from random import randint
from typing import Tuple

import torch
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.generation.utils import GenerationConfig


def init_model(base: str, output: str) -> Tuple[PeftModelForCausalLM, PreTrainedTokenizer]:
    """加载模型和词表"""
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=output,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        pretrained_model_name=base
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=base,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def predict() -> None:
    """预测主方法"""
    parser = ArgumentParser()
    parser.add_argument("--base", "-b", help="absolute path for base model")
    parser.add_argument("--output", "-o", help="absolute path for fine-tuned model")
    args = parser.parse_args()
    model, tokenizer = init_model(base=args.base, output=args.output)
    common = load(fp=open(file="data/predict.json", encoding="utf-8"))
    professional = load(fp=open(file="data/train.json", encoding="utf-8"))
    n = len(professional) - 1
    with open(file="/tmp/output/prediction.txt", mode="w+", encoding="utf-8") as f:
        f.write("===================================通识问答===================================")
        f.write("\n\n")
        for question in common:
            answer = model.chat(tokenizer, [{"role": "user", "content": question}])
            f.write("Question:{}\n".format(question))
            f.write("Answer:{}\n".format(answer))
            f.write("\n")
            if torch.backends.mps.is_available():  # noqa
                torch.mps.empty_cache()  # noqa
        f.write("===================================专业问答===================================")
        f.write("\n\n")
        for _ in range(50):
            conversation = professional[randint(0, n)]["conversations"]
            for i in range(len(conversation) >> 1):
                question, reference = conversation[i << 1]["value"], conversation[(i << 1) + 1]["value"]
                answer = model.chat(tokenizer, [{"role": "user", "content": question}])
                f.write("Question:{}\n".format(question))
                f.write("Answer:{}\n".format(answer))
                f.write("Reference:{}\n".format(reference))
                f.write("\n")
                if torch.backends.mps.is_available():  # noqa
                    torch.mps.empty_cache()  # noqa
    return


if __name__ == "__main__":
    predict()
