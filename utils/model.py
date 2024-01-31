from os import environ
from threading import Thread
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 判断是否使用NPU资源，默认使用GPU资源
use_gpu = True
if not torch.cuda.is_available():
    import torch_npu  # noqa

    use_gpu = False


class LargeLanguageModel(object):
    """base class for all large language models"""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


class ChatModel(LargeLanguageModel):
    """base class for chat models"""

    def __init__(self, name: str, path: str):
        super(ChatModel, self).__init__(name=name, path=path)
        self.model = None
        self.tokenizer = None

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        """生成模型答复文本"""
        raise NotImplementedError("method: generate")

    def stream(self, conversation: List[Dict[str, str]]):
        """流式生成模型答复，使用字符模式"""
        raise NotImplementedError("method: stream")


class CompletionModel(LargeLanguageModel):
    """base class for completion models"""

    def __init__(self, name: str, path: str):
        super(CompletionModel, self).__init__(name=name, path=path)
        self.model = None
        self.tokenizer = None

    def generate(self, question: str) -> str:
        """生成模型填充文本"""
        raise NotImplementedError("method: generate")

    def stream(self, question: str):
        """流式生成模型填充，使用字符模式"""
        raise NotImplementedError("method: stream")


class EmbeddingModel(LargeLanguageModel):
    """base class for embedding models"""

    def __init__(self, name: str, path: str):
        super(EmbeddingModel, self).__init__(name=name, path=path)
        self.model = None

    def embedding(self, sentence: str) -> List[float]:
        """生成模型嵌入结果"""
        raise NotImplementedError("method: embedding")


class Baichuan2ChatModel(ChatModel):  # noqa
    """class for baichuan2 chat model"""

    def __init__(self, name: str, path: str):
        super(Baichuan2ChatModel, self).__init__(name=name, path=path)
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


class CodefuseDeepseekModel(ChatModel):  # noqa
    """class for codefuse deepseek model"""

    def __init__(self, name: str, path: str):
        super(CodefuseDeepseekModel, self).__init__(name=name, path=path)
        from auto_gptq import AutoGPTQForCausalLM
        environ["TOKENIZERS_PARALLELISM"] = "false"  # noqa
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path=self.path,
            device_map="cuda:0",  # noqa
            inject_fused_attention=False,
            inject_fused_mlp=False,
            use_cuda_fp16=True,
            use_safetensors=True,
            disable_exllama=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.path,
            use_fast=False,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")

    def generate(self, conversation: List[Dict[str, str]]) -> str:
        pass

    def stream(self, conversation: List[Dict[str, str]]):
        pass

    def stream_task(self, input_ids, streamer: TextIteratorStreamer) -> None:
        """流式响应的子任务"""
        pass

    @staticmethod
    def chat_template(conversation: List[Dict[str, str]]) -> str:
        """生成提示词模板"""
        ans = ""
        for message in conversation:
            role, content = message["role"], message["content"]
            if role == "user":
                ans += "<s>human\n{}\n<s>bot\n".format(content)
            elif role == "assistant":
                ans += "{}<｜end▁of▁sentence｜>\n".format(content)
            else:
                raise ValueError("error conversation or history")
        return ans


class DeepseekCoderInstructModel(ChatModel):  # noqa
    """class for deepseek coder instruct model"""

    def __init__(self, name: str, path: str):
        super(DeepseekCoderInstructModel, self).__init__(name=name, path=path)
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


class Internlm2ChatModel(ChatModel):  # noqa
    """class for internlm2 chat model"""

    def __init__(self, name: str, path: str):
        super(Internlm2ChatModel, self).__init__(name=name, path=path)
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

    def stream_task(self, input_ids, streamer: TextIteratorStreamer) -> None:
        """流式响应的子任务"""
        pass


class SusChatModel(ChatModel):
    """class for sus chat model"""

    def __init__(self, name: str, path: str):
        super(SusChatModel, self).__init__(name=name, path=path)
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

    def stream_task(self, input_ids, streamer: TextIteratorStreamer) -> None:
        """流式响应的子任务"""
        pass

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


class QwenModel(CompletionModel):  # noqa
    """class for qwen model"""

    def __init__(self, name: str, path: str):
        super(QwenModel, self).__init__(name=name, path=path)
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


class BceModel(EmbeddingModel):
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


class M3eModel(EmbeddingModel):
    """class for m3e model"""

    def __init__(self, name: str, path: str):
        super(M3eModel, self).__init__(name=name, path=path)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name_or_path=self.path, device="cuda:0")  # noqa

    def embedding(self, sentence: str) -> List[float]:
        result = self.model.encode(sentences=sentence)
        return result.tolist()
