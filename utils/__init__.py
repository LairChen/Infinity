from .config import *
from .model import *
from .schema import *

MODEL_TYPE_DICT = {
    "Baichuan2-7B-Chat": BaichuanModel,  # noqa
    "Baichuan2-13B-Chat": BaichuanModel,  # noqa
    "deepseek-coder-6.7b-instruct": DeepseekModel,  # noqa
    "m3e-large": M3eModel
}

__all__ = [s for s in dir() if not s.startswith("_")]
