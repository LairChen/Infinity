from abc import ABC
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型常量
CONTENT_MAX_LENGTH = 4096
EMBEDDING_MAX_LENGTH = 1536


class BaseModel(object):
    """base class for all models"""

    def __init__(self):
        self.name = "Infinity"
        self.model = None
        self.tokenizer = None

    def finetune(self):  # noqa
        """模型微调"""
        raise NotImplementedError("method: train")

    def generate(self):
        """生成模型答复文本"""
        raise NotImplementedError("method: generate")

    def stream(self, conversation: List[Dict[str, str]]):
        """流式生成模型答复，使用para模式"""
        raise NotImplementedError("method: stream")

    def embedding(self, content: str):
        """生成模型嵌入结果"""
        raise NotImplementedError("method: embedding")


class BaichuanModel(BaseModel, ABC):  # noqa
    """class for baichuan"""

    def __init__(self, name: str, path: str):
        super().__init__()
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path,
            use_fast=False,
            trust_remote_code=True
        )

    def stream(self, conversation: List[Dict[str, str]]):
        for answer in self.model.chat(self.tokenizer, conversation, stream=True):
            if torch.backends.mps.is_available():  # noqa
                torch.mps.empty_cache()  # noqa
            if len(answer) >= CONTENT_MAX_LENGTH:
                break
            yield answer


class DeepseekModel(BaseModel, ABC):  # noqa
    """class for deepseek"""

    def __init__(self):
        super().__init__()


class M3eModel(BaseModel, ABC):
    """class for m3e"""

    def __init__(self):
        super().__init__()
