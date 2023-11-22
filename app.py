from json import dumps
from os import system, getenv
from time import time
from typing import Dict, Union
from uuid import uuid4

import gradio as gr
import torch
from fastapi import FastAPI
from flasgger import Schema, fields
from marshmallow import validate
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig


# 使用marshmallow作序列化和参数校验
class ChatDeltaSchema(Schema):
    role = fields.Str()
    content = fields.Str()


class ChatMessageSchema(Schema):
    role = fields.Str(required=True)
    content = fields.Str(required=True)


class CreateChatCompletionSchema(Schema):
    model = fields.Str(required=True, metadata={"example": "baichuan2-7b-chat"})  # noqa
    messages = fields.List(fields.Nested(nested=ChatMessageSchema), required=True)  # noqa
    max_tokens = fields.Int(load_default=None)
    n = fields.Int(load_default=1)
    temperature = fields.Float(load_default=1.0)
    top_p = fields.Float(load_default=1.0)
    frequency_penalty = fields.Float(load_default=0.0)
    presence_penalty = fields.Float(load_default=0.0)
    stream = fields.Bool(load_default=False)


class ChatCompletionChunkChoiceSchema(Schema):
    index = fields.Int()
    delta = fields.Nested(nested=ChatDeltaSchema)  # noqa
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),  # noqa
        metadata={"example": "stop"})


class ChatCompletionChoiceSchema(Schema):
    index = fields.Int()
    message = fields.Nested(nested=ChatMessageSchema)  # noqa
    finish_reason = fields.Str(
        validate=validate.OneOf(choices=["stop", "length", "content_filter", "function_call"]),  # noqa
        metadata={"example": "stop"})


class ChatCompletionChunkSchema(Schema):
    id = fields.Str(dump_default=lambda: uuid4().hex)
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChunkChoiceSchema))  # noqa
    object = fields.Constant(constant="chat.completion.chunk")
    created = fields.Int(dump_default=lambda: int(time()))


class ChatCompletionSchema(Schema):
    id = fields.Str(dump_default=lambda: uuid4().hex)
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChoiceSchema))  # noqa
    object = fields.Constant(constant="chat.completion")
    created = fields.Int(dump_default=lambda: int(time()))


def init_env() -> None:
    system("mkdir /tmp/dataset")
    system("unzip /dataset/Baichuan2-7B-Chat.zip -d /tmp/dataset")
    return


def init_model():
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="/pretrainmodel",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        pretrained_model_name="/tmp/dataset/Baichuan2-7B-Chat"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/tmp/dataset/Baichuan2-7B-Chat",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer




def sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)



def create_chat_completion():
    """Chat接口"""
    return 213




app = FastAPI()
demo = gr.Interface(
    fn=create_chat_completion,
    inputs=gr.components.Textbox(label="Inputs"),
    outputs=gr.components.Textbox(label="Outputs"),
    allow_flagging="never"
)
app = gr.mount_gradio_app(app=app, blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
