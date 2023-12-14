from abc import ABC


class BaseModel(object):
    """所有模型的基类"""

    def __init__(self):
        self.name = ""
        self.model = None
        self.tokenizer = None

    def finetune(self):  # noqa
        """模型微调"""
        raise NotImplementedError("method: train")

    def generate(self):
        """生成模型答复文本"""
        raise NotImplementedError("method: generate")

    def stream(self):
        """流式生成模型答复，使用para模式"""
        raise NotImplementedError("method: stream")

    def embedding(self):
        """生成模型嵌入结果"""
        raise NotImplementedError("method: embedding")


class BaichuanModel(BaseModel, ABC):  # noqa
    def __init__(self):
        super().__init__()


class DeepseekModel(BaseModel, ABC):  # noqa
    def __init__(self):
        super().__init__()


class M3eModel(BaseModel, ABC):
    def __init__(self):
        super().__init__()
