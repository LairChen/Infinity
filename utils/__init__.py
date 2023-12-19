from .config import *
from .model import *
from .schema import *

CHAT_MODEL_TYPE = {
    "Baichuan2-7B-Chat": BaichuanModel,  # noqa
    "Baichuan2-13B-Chat": BaichuanModel,  # noqa
    "deepseek-coder-1.3b-instruct": DeepseekModel,  # noqa
    "deepseek-coder-6.7b-instruct": DeepseekModel,  # noqa
    "deepseek-coder-33b-instruct": DeepseekModel,  # noqa
    "deepseek-coder-33B-instruct-GPTQ": DeepseekGPTQModel  # noqa
}

EMBEDDINGS_MODEL_TYPE = {
    "m3e-base": M3eModel,
    "m3e-small": M3eModel,
    "m3e-large": M3eModel
}

__all__ = [s for s in dir() if not s.startswith("_")]
