from json import dumps
from os import system, getenv
from time import time
from typing import Dict, Tuple, Union
from uuid import uuid4

import gradio as gr
import torch
from fastapi import FastAPI
from flasgger import Schema, fields
from flask import Flask, Blueprint, Response, current_app, request, stream_with_context
from flask_cors import CORS
from marshmallow import validate
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig

# 使用marshmallow作序列化和参数校验
# blueprint = Blueprint(name="Chat", import_name=__name__, url_prefix="/v1/chat")  # 声明蓝图


class ChatMessageSchema(Schema):
    role = fields.Str(required=True)
    content = fields.Str(required=True)


class ChatDeltaSchema(Schema):
    role = fields.Str()
    content = fields.Str()


class CreateChatCompletionSchema(Schema):
    model = fields.Str(required=True, metadata={"example": "baichuan2-7b-chat"})  # noqa
    messages = fields.List(fields.Nested(nested=ChatMessageSchema), required=True)  # noqa
    max_tokens = fields.Int(load_default=None)
    temperature = fields.Float(load_default=1.0)
    top_p = fields.Float(load_default=1.0)
    n = fields.Int(load_default=1)
    stream = fields.Bool(load_default=False)
    presence_penalty = fields.Float(load_default=0.0)
    frequency_penalty = fields.Float(load_default=0.0)


class ChatCompletionChunkChoiceSchema(Schema):
    index = fields.Int()
    delta = fields.Nested(nested=ChatDeltaSchema)  # noqa
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),  # noqa
        metadata={"example": "stop"})


class ChatCompletionChunkSchema(Schema):
    id = fields.Str(dump_default=lambda: uuid4().hex)
    object = fields.Constant(constant="chat.completion.chunk")
    created = fields.Int(dump_default=lambda: int(time()))
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChunkChoiceSchema))  # noqa


class ChatCompletionChoiceSchema(Schema):
    index = fields.Int()
    message = fields.Nested(nested=ChatMessageSchema)  # noqa
    finish_reason = fields.Str(
        validate=validate.OneOf(choices=["stop", "length", "content_filter", "function_call"]),  # noqa
        metadata={"example": "stop"})


class ChatCompletionSchema(Schema):
    id = fields.Str(dump_default=lambda: uuid4().hex)
    object = fields.Constant(constant="chat.completion")
    created = fields.Int(dump_default=lambda: int(time()))
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChoiceSchema))  # noqa


def init_env() -> None:
    system("mkdir /tmp/dataset")
    system("unzip /dataset/Baichuan2-7B-Chat.zip -d /tmp/dataset")
    # system("chmod +x frpc/frpc")  # noqa
    # system("nohup ./frpc/frpc -c frpc/frpc.ini &")  # noqa
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


# def init_app() -> Tuple[Flask, Blueprint]:
#     """创建接口服务"""
#     app = Flask(__name__)  # 声明主服务
#     CORS(app=app)  # 允许跨域
#
#     app.register_blueprint(blueprint=blueprint)  # 注册蓝图
#
#     @app.after_request
#     def after_request(resp: Response) -> Response:
#         """请求后处理"""
#         if torch.backends.mps.is_available():  # noqa
#             torch.mps.empty_cache()  # noqa
#         return resp
#
#     return app, blueprint


init_env()
my_model, my_tokenizer = init_model()


def sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


@stream_with_context
def stream_chat_generate(messages):
    """Chat流式"""
    delta = ChatDeltaSchema().dump({"role": "assistant"})
    choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
    yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    position = 0
    for response in my_model.chat(my_tokenizer, messages, stream=True):
        content = response[position:]
        if not content:
            continue
        if torch.backends.mps.is_available():  # noqa
            torch.mps.empty_cache()  # noqa
        delta = ChatDeltaSchema().dump({"content": content})
        choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
        yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
        position = len(response)
    choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": {}, "finish_reason": "stop"})
    yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    yield sse(line="[DONE]")


# @blueprint.route("/completions", methods=["POST"])
def create_chat_completion():
    """Chat接口"""
    chat_dict = CreateChatCompletionSchema().load(request.json)
    # if chat_dict["stream"]:
    #     # 切换到流式
    #     return current_app.response_class(response=stream_chat_generate(chat_dict["messages"]), mimetype="text/event-stream")
    response = my_model.chat(my_tokenizer, chat_dict["messages"])
    message = ChatMessageSchema().dump({"role": "assistant", "content": response})
    choice = ChatCompletionChoiceSchema().dump({"index": 0, "message": message, "finish_reason": "stop"})
    return ChatCompletionSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]})  # noqa


if __name__ == "__main__":
    # my_app, _ = init_app()  # noqa
    # my_app.run(host="0.0.0.0", port=8262, debug=False)

    app = FastAPI()
    demo = gr.Interface(
        fn=create_chat_completion,
        inputs=gr.components.Textbox(label="Inputs"),
        outputs=gr.components.Textbox(label="Outputs"),
        allow_flagging="never"
    )
    app = gr.mount_gradio_app(app, demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
