from abc import ABC
from threading import Thread
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 模型常量
MAX_CONTENT_LENGTH = 4096
MAX_EMBEDDING_LENGTH = 1536  # OpenAI API 嵌入维度标准 1536


class BaseModel(object):
    """base class for all models"""

    def __init__(self):
        self.name = "Infinity"
        self.model = None
        self.tokenizer = None

    def finetune(self):  # noqa
        """模型微调"""
        raise NotImplementedError("method: train")

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        """生成模型答复文本"""
        raise NotImplementedError("method: generate")

    def stream(self, conversation: List[Dict[str, str]]):
        """流式生成模型答复，使用para模式"""
        raise NotImplementedError("method: stream")

    def embedding(self, sentence: List[str]) -> List[List[float]]:
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

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        return self.model.chat(self.tokenizer, conversation)

    def stream(self, conversation: List[Dict[str, str]]):
        for answer in self.model.chat(self.tokenizer, conversation, stream=True):
            if torch.backends.mps.is_available():  # noqa
                torch.mps.empty_cache()  # noqa
            if len(answer) >= MAX_CONTENT_LENGTH:
                break
            yield answer


class DeepseekModel(BaseModel, ABC):  # noqa
    """class for deepseek"""

    def __init__(self, name: str, path: str):
        super().__init__()
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path,
            use_fast=False,
            trust_remote_code=True
        )

    def stream(self, conversation: List[Dict[str, str]]):
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "do_sample": False,
            "max_new_tokens": 1024,
            "num_beams": 1,
            "repetition_penalty": 1,
            "eos_token_id": 32021
        }
        Thread(target=self.model.generate, kwargs=generate_kwargs).start()
        answer = ""
        for text in streamer:
            answer += text.replace("<|EOT|>", "")
            if torch.backends.mps.is_available():  # noqa
                torch.mps.empty_cache()  # noqa
            if len(answer) >= MAX_CONTENT_LENGTH:
                break
            yield answer


class DeepseekGPTQModel(BaseModel, ABC):  # noqa
    """class for deepseek GPTQ"""

    def __init__(self, name: str, path: str):
        super().__init__()
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path,
            use_fast=False,
            trust_remote_code=True
        )

    def stream(self, conversation: List[Dict[str, str]]):
        template = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, " \
                   "and you only answer questions related to computer science. " \
                   "For politically sensitive questions, security and privacy issues, and other non-computer science questions, " \
                   "you will refuse to answer.\n" \
                   "### Instruction:\n" \
                   f"{conversation[-1]['content']}\n" \
                   "### Response:\n"
        input_ids = self.tokenizer(template, return_tensors="pt").input_ids.to(self.model.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "do_sample": False,
            "max_new_tokens": 1024,
            "num_beams": 1,
            "repetition_penalty": 1,
            "eos_token_id": 32021
        }
        Thread(target=self.model.generate, kwargs=generate_kwargs).start()
        answer = ""
        for text in streamer:
            answer += text.replace("<|EOT|>", "")
            if torch.backends.mps.is_available():  # noqa
                torch.mps.empty_cache()  # noqa
            if len(answer) >= MAX_CONTENT_LENGTH:
                break
            yield answer


class M3eModel(BaseModel, ABC):
    """class for m3e"""

    def __init__(self, name: str, path: str):
        super().__init__()
        self.name = name
        self.model = SentenceTransformer(model_name_or_path=path)

    def embedding(self, sentence: str) -> List[float]:
        result = self.model.encode(sentences=sentence)
        if len(result) < MAX_EMBEDDING_LENGTH:
            result = PolynomialFeatures(degree=2).fit_transform(X=result.reshape(1, -1)).flatten()
            if len(result) < MAX_EMBEDDING_LENGTH:
                result = np.pad(array=result, pad_width=(0, MAX_EMBEDDING_LENGTH - len(result)))
            else:
                result = result[:MAX_EMBEDDING_LENGTH]
        else:
            result = result[:MAX_EMBEDDING_LENGTH]
        result = result / np.linalg.norm(x=result)
        return result.tolist()
