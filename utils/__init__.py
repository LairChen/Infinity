from os import listdir
from re import match
from typing import Union, Optional

from .config import appHost, appPort, path_train_pretrain, path_train_finetune, path_eval_pretrain, path_eval_finetune, llm
from .model import Baichuan2ChatModel, CodefuseDeepseekModel, DeepseekCoderInstructModel, Internlm2ChatModel, SusChatModel
from .model import BceModel, M3eModel
from .model import ChatModel, CompletionModel, EmbeddingModel
from .model import QwenModel
from .schema import ChatMessageSchema, ChatRequestSchema, ChatChoiceSchema, ChatChoiceChunkSchema, ChatResponseSchema, \
    ChatResponseChunkSchema
from .schema import CompletionsRequestSchema
from .schema import EmbeddingsRequestSchema, EmbeddingsDataSchema, EmbeddingsUsageSchema, EmbeddingsResponseSchema

CHAT_MODEL_TYPE = {
    "Baichuan2-7B-Chat": Baichuan2ChatModel,  # noqa
    "Baichuan2-13B-Chat": Baichuan2ChatModel,  # noqa
    "CodeFuse-DeepSeek-33B-4bits": CodefuseDeepseekModel,
    "deepseek-coder-1.3b-instruct": DeepseekCoderInstructModel,  # noqa
    "deepseek-coder-6.7b-instruct": DeepseekCoderInstructModel,  # noqa
    "deepseek-coder-33B-instruct-GPTQ": DeepseekCoderInstructModel,  # noqa
    "internlm2-chat-7b": Internlm2ChatModel,  # noqa
    "SUS-Chat-34B-GPTQ": SusChatModel  # noqa
}

COMPLETION_MODEL_TYPE = {
    "Qwen-1.8B": QwenModel,  # noqa
    "Qwen-7B": QwenModel,  # noqa
    "Qwen-14B": QwenModel  # noqa
}

EMBEDDING_MODEL_TYPE = {
    "bce-embedding-base-v1": BceModel,
    "m3e-base": M3eModel,
    "m3e-small": M3eModel,
    "m3e-large": M3eModel
}


def init_language_model() -> Union[ChatModel, CompletionModel]:
    """初始化对话/补全模型"""
    with open(file="{}/model_type.txt".format(path_eval_finetune), mode="r", encoding="utf-8") as f:
        my_model_name = f.read().strip()
    if CHAT_MODEL_TYPE.get(my_model_name, None) is not None:
        my_model = CHAT_MODEL_TYPE[my_model_name](name=my_model_name, path=path_eval_finetune)
    elif COMPLETION_MODEL_TYPE.get(my_model_name, None) is not None:
        my_model = COMPLETION_MODEL_TYPE[my_model_name](name=my_model_name, path=path_eval_finetune)
    else:
        raise FileNotFoundError("no existing language model")
    return my_model


def init_embedding_model() -> Optional[EmbeddingModel]:
    """初始化嵌入模型"""
    for filename in listdir(path_eval_pretrain):
        modelname = match(pattern="(.*)\.zip", string=filename)  # noqa
        if modelname is not None:
            my_model_name = modelname.groups()[0]
            break
    else:
        return None
    my_model = EMBEDDING_MODEL_TYPE[my_model_name](name=my_model_name, path=my_model_name)
    return my_model


__all__ = [s for s in dir() if not s.startswith("_")]
