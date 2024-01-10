from .config import *
from .model import *
from .schema import *

CHAT_MODEL_TYPE = {
    "Baichuan2-7B-Chat": Baichuan2Model,  # noqa
    "Baichuan2-13B-Chat": Baichuan2Model,  # noqa
    "deepseek-coder-1.3b-instruct": DeepseekModel,  # noqa
    "deepseek-coder-6.7b-instruct": DeepseekModel,  # noqa
    "deepseek-coder-33b-instruct": DeepseekModel,  # noqa
    "deepseek-coder-33B-instruct-GPTQ": DeepseekModel,  # noqa
    "SUS-Chat-34B": SusModel,
    "SUS-Chat-34B-GPTQ": SusModel  # noqa
}

COMPLETION_MODEL_TYPE = {

}

EMBEDDING_MODEL_TYPE = {
    "m3e-base": M3eModel,
    "m3e-small": M3eModel,
    "m3e-large": M3eModel
}

__all__ = [s for s in dir() if not s.startswith("_")]
