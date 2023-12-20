from threading import Thread
from typing import Dict, List

import numpy as np
import torch
from auto_gptq import exllama_set_max_input_length
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class BaseModel(object):
    """base class for all models"""

    def __init__(self, name: str):
        self.name = name


class BaseChatModel(BaseModel):
    """base class for chat models"""

    def __init__(self, name: str):
        super(BaseChatModel, self).__init__(name=name)
        self.model = None
        self.tokenizer = None

    def finetune(self):  # noqa
        """模型微调"""
        raise NotImplementedError("method: train")

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        """生成模型答复文本"""
        raise NotImplementedError("method: generate")

    def stream(self, conversation: List[Dict[str, str]]):
        """流式生成模型答复，使用字符模式"""
        raise NotImplementedError("method: stream")


class BaseEmbeddingsModel(BaseModel):
    """base class for embeddings models"""

    def __init__(self, name: str):
        super(BaseEmbeddingsModel, self).__init__(name=name)
        self.model = None

    def embedding(self, sentence: List[str]) -> List[List[float]]:
        """生成模型嵌入结果"""
        raise NotImplementedError("method: embedding")


class BaichuanModel(BaseChatModel):  # noqa
    """class for baichuan model"""

    def __init__(self, name: str, path: str):
        super(BaichuanModel, self).__init__(name=name)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path,
            use_fast=False,
            trust_remote_code=True
        )
        self.model = self.model.eval()

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        return self.model.chat(self.tokenizer, conversation)

    def stream(self, conversation: List[Dict[str, str]]):
        position = 0
        for answer in self.model.chat(self.tokenizer, conversation, stream=True):
            if torch.backends.mps.is_available():  # noqa
                torch.mps.empty_cache()  # noqa
            yield answer[position:]
            position = len(answer)


class DeepseekModel(BaseChatModel):  # noqa
    """class for deepseek model"""

    def __init__(self, name: str, path: str):
        super(DeepseekModel, self).__init__(name=name)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path,
            use_fast=False,
            trust_remote_code=True
        )
        # GPTQ量化模型需要额外扩展输入文本的长度
        if self.name.endswith("GPTQ"):  # noqa
            self.model = exllama_set_max_input_length(model=self.model, max_input_length=4096)

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(inputs=input_ids, do_sample=False, eos_token_id=32021, max_new_tokens=1024)
        return self.tokenizer.decode(token_ids=output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

    def stream(self, conversation: List[Dict[str, str]]):
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, timeout=5.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "do_sample": False,
            "eos_token_id": 32021,
            "max_new_tokens": 1024,
            "num_beams": 1,
            "repetition_penalty": 1
        }
        Thread(target=self.model.generate, kwargs=generate_kwargs).start()
        for text in streamer:
            if torch.backends.mps.is_available():  # noqa
                torch.mps.empty_cache()  # noqa
            yield text.replace("<|EOT|>", "")


class M3eModel(BaseEmbeddingsModel):
    """class for m3e model"""

    def __init__(self, name: str, path: str):
        super(M3eModel, self).__init__(name=name)
        self.model = SentenceTransformer(model_name_or_path=path)

    def embedding(self, sentence: str) -> List[float]:
        result = self.model.encode(sentences=sentence)
        # OpenAI API 嵌入维度标准 1536
        if len(result) < 1536:
            result = PolynomialFeatures(degree=2).fit_transform(X=result.reshape(1, -1)).flatten()
            if len(result) < 1536:
                result = np.pad(array=result, pad_width=(0, 1536 - len(result)))
            else:
                result = result[:1536]
        else:
            result = result[:1536]
        result = result / np.linalg.norm(x=result)
        return result.tolist()
