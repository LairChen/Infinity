from .config import appHost, appPort, path_train_pretrain, path_train_finetune, path_eval_pretrain, path_eval_finetune, llm
from .model import *
from .schema import *

CHAT_MODEL_TYPE = {
    "Baichuan2-7B-Chat": Baichuan2Model,  # noqa
    "Baichuan2-13B-Chat": Baichuan2Model,  # noqa
    "deepseek-coder-1.3b-instruct": DeepseekModel,  # noqa
    "deepseek-coder-6.7b-instruct": DeepseekModel,  # noqa
    "deepseek-coder-33B-instruct-GPTQ": DeepseekModel,  # noqa
    "SUS-Chat-34B-GPTQ": SusModel  # noqa
}

COMPLETION_MODEL_TYPE = {
    #"Qwen-1.8B": QwenBaseModel,  # noqa
    #"Qwen-7B": QwenBaseModel  # noqa
}

EMBEDDING_MODEL_TYPE = {
    "bce-embedding-base-v1": BceModel,
    "m3e-base": M3eModel,
    "m3e-small": M3eModel,
    "m3e-large": M3eModel
}

__all__ = [s for s in dir() if not s.startswith("_")]
