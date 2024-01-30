from threading import Thread
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 判断是否使用NPU资源，默认使用GPU资源
use_gpu = True
if not torch.cuda.is_available():
    import torch_npu  # noqa

    use_gpu = False


class BaseModel(object):
    """base class for all models"""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


class BaseChatModel(BaseModel):
    """base class for chat models"""

    def __init__(self, name: str, path: str):
        super(BaseChatModel, self).__init__(name=name, path=path)
        self.model = None
        self.tokenizer = None

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        """生成模型答复文本"""
        raise NotImplementedError("method: generate")

    def stream(self, conversation: List[Dict[str, str]]):
        """流式生成模型答复，使用字符模式"""
        raise NotImplementedError("method: stream")


class BaseCompletionModel(BaseModel):
    """base class for completion models"""

    def __init__(self, name: str, path: str):
        super(BaseCompletionModel, self).__init__(name=name, path=path)
        self.model = None
        self.tokenizer = None

    def generate(self, question: str) -> str:
        """生成模型填充文本"""
        raise NotImplementedError("method: generate")

    def stream(self, question: str):
        """流式生成模型填充，使用字符模式"""
        raise NotImplementedError("method: stream")


class BaseEmbeddingModel(BaseModel):
    """base class for embedding models"""

    def __init__(self, name: str, path: str):
        super(BaseEmbeddingModel, self).__init__(name=name, path=path)
        self.model = None

    def embedding(self, sentence: str) -> List[float]:
        """生成模型嵌入结果"""
        raise NotImplementedError("method: embedding")


class Baichuan2Model(BaseChatModel):  # noqa
    """class for baichuan2 model"""

    def __init__(self, name: str, path: str):
        super(Baichuan2Model, self).__init__(name=name, path=path)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.path,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # noqa
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.path,
            use_fast=False,
            trust_remote_code=True
        )

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        return self.model.chat(tokenizer=self.tokenizer, messages=conversation)

    def stream(self, conversation: List[Dict[str, str]]):
        position = 0
        for answer in self.model.chat(tokenizer=self.tokenizer, messages=conversation, stream=True):
            yield answer[position:]
            position = len(answer)


class DeepseekModel(BaseChatModel):  # noqa
    """class for deepseek model"""

    def __init__(self, name: str, path: str):
        super(DeepseekModel, self).__init__(name=name, path=path)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.path,
            torch_dtype=torch.float16,
            device_map="cuda:0" if use_gpu else "npu:0",  # noqa
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.path,
            use_fast=False,
            trust_remote_code=True
        )
        # GPTQ量化模型需要额外扩展输入文本的长度
        if self.name.endswith("GPTQ"):  # noqa
            from auto_gptq import exllama_set_max_input_length
            self.model = exllama_set_max_input_length(model=self.model, max_input_length=4096)

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        if not use_gpu:
            # NPU任务需要显式的声明设备
            torch_npu.npu.set_device("npu:0")  # noqa
        input_ids = self.tokenizer.apply_chat_template(conversation=conversation, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(inputs=input_ids, do_sample=True, eos_token_id=32021, max_new_tokens=4096)
        return self.tokenizer.decode(token_ids=output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

    def stream(self, conversation: List[Dict[str, str]]):
        if not use_gpu:
            # NPU任务需要显式的声明设备
            torch_npu.npu.set_device("npu:0")  # noqa
        input_ids = self.tokenizer.apply_chat_template(conversation=conversation, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
        Thread(target=self.stream_task, args=(input_ids, streamer)).start()
        for text in streamer:
            yield text

    def stream_task(self, input_ids: torch.Tensor, streamer: TextIteratorStreamer) -> None:
        """流式响应的子任务"""
        if not use_gpu:
            # NPU任务需要显式的声明设备
            torch_npu.npu.set_device("npu:0")  # noqa
        self.model.generate(inputs=input_ids, streamer=streamer, do_sample=True, eos_token_id=32021, max_new_tokens=4096)
        return None


class Internlm2Model(BaseChatModel):  # noqa
    """class for internlm model"""

    def __init__(self, name: str, path: str):
        super(Internlm2Model, self).__init__(name=name, path=path)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.path,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # noqa
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.path,
            use_fast=False,
            trust_remote_code=True
        )

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        pass

    def stream(self, conversation: List[Dict[str, str]]):
        query = conversation[-1]["content"]
        history = [(conversation[i << 1]["content"], conversation[(i << 1) + 1]["content"])
                   for i in range(len(conversation) >> 1)]
        position = 0
        for answer, _ in self.model.stream_chat(tokenizer=self.tokenizer, query=query, history=history):
            yield answer[position:]
            position = len(answer)


class SusModel(BaseChatModel):
    """class for sus model"""

    def __init__(self, name: str, path: str):
        super(SusModel, self).__init__(name=name, path=path)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.path,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # noqa
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.path,
            use_fast=False,
            trust_remote_code=True
        )
        # GPTQ量化模型需要额外扩展输入文本的长度
        if self.name.endswith("GPTQ"):  # noqa
            from auto_gptq import exllama_set_max_input_length
            self.model = exllama_set_max_input_length(model=self.model, max_input_length=4096)

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        input_ids = self.tokenizer.encode(text=self.chat_template(conversation), return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(inputs=input_ids, do_sample=True, max_new_tokens=4096)
        return self.tokenizer.decode(token_ids=output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

    def stream(self, conversation: List[Dict[str, str]]):
        input_ids = self.tokenizer.encode(text=self.chat_template(conversation), return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, timeout=5.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "do_sample": True,
            "max_new_tokens": 4096
        }
        Thread(target=self.model.generate, kwargs=generate_kwargs).start()
        for text in streamer:
            yield text

    @staticmethod
    def chat_template(conversation: List[Dict[str, str]]) -> str:
        """生成提示词模板"""
        ans = ""
        for message in conversation:
            role, content = message["role"], message["content"]
            if role == "user":
                ans += "### Human: {}\n\n### Assistant: ".format(content)
            elif role == "assistant":
                ans += "{}\n\n".format(content)
            else:
                raise ValueError("error conversation or history")
        return ans


class QwenBaseModel(BaseCompletionModel):  # noqa
    """class for qwen base model"""

    def __init__(self, name: str, path: str):
        super(QwenBaseModel, self).__init__(name=name, path=path)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.path,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # noqa
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.path,
            use_fast=False,
            trust_remote_code=True
        )

    def generate(self, question: str) -> str:
        input_ids = self.tokenizer(question, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**input_ids, do_sample=True, max_new_tokens=4096)
        return self.tokenizer.decode(token_ids=output_ids[0], skip_special_tokens=True)

    def stream(self, question: str):
        input_ids = self.tokenizer(question, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
        Thread(target=self.stream_task, args=(input_ids, streamer)).start()
        for text in streamer:
            yield text

    def stream_task(self, input_ids: Dict, streamer: TextIteratorStreamer) -> None:
        """流式响应的子任务"""
        self.model.generate(**input_ids, streamer=streamer, do_sample=True, max_new_tokens=4096)
        return None


class BceModel(BaseEmbeddingModel):
    """class for bce model"""

    def __init__(self, name: str, path: str):
        super(BceModel, self).__init__(name=name, path=path)
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.path,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # noqa
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.path,
            use_fast=False,
            trust_remote_code=True
        )

    def embedding(self, sentence: str) -> List[float]:
        input_ids = self.tokenizer([sentence], padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = {k: v.to(self.model.device) for k, v in input_ids.items()}
        output_ids = self.model(**input_ids, return_dict=True)
        output_ids = output_ids.last_hidden_state[:, 0]
        output_ids = output_ids / output_ids.norm(dim=1, keepdim=True)
        return output_ids.tolist()[0]


class M3eModel(BaseEmbeddingModel):
    """class for m3e model"""

    def __init__(self, name: str, path: str):
        super(M3eModel, self).__init__(name=name, path=path)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name_or_path=self.path, device="cuda:0")  # noqa

    def embedding(self, sentence: str) -> List[float]:
        result = self.model.encode(sentences=sentence)
        return result.tolist()
