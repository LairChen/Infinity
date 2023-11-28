from configparser import ConfigParser
from json import dumps
from random import uniform
from time import time, sleep
from typing import Dict, List, Union
from uuid import uuid4

from flasgger import Schema, fields
from flask import Flask, Blueprint, current_app, request, stream_with_context
from flask_cors import CORS
from marshmallow import validate
from requests import post

cfg = ConfigParser()
cfg.read(filenames="conf/config.ini", encoding="utf-8")
appHost = cfg.get(section="app", option="app.host")
appPort = cfg.getint(section="app", option="app.port")
baseURL = cfg.get(section="app", option="app.gradio.addr")  # noqa
blueprint = Blueprint(name="Chat", import_name=__name__, url_prefix="/v1/chat")


# 使用marshmallow做序列化和参数校验

class ChatMessageSchema(Schema):
    """Chat消息结构映射"""
    role = fields.Str(required=True)
    content = fields.Str(required=True)


class ChatDeltaSchema(Schema):
    """Chat流式结构映射"""
    role = fields.Str()
    content = fields.Str()


class CreateChatCompletionSchema(Schema):
    """Chat接口请求数据结构解析"""
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"}, required=True)  # noqa
    messages = fields.List(fields.Nested(nested=ChatMessageSchema), required=True)  # noqa
    stream = fields.Bool(load_default=False)
    max_tokens = fields.Int(load_default=None)
    n = fields.Int(load_default=1)
    seed = fields.Int(load_default=1)
    top_p = fields.Float(load_default=1.0)
    temperature = fields.Float(load_default=1.0)
    presence_penalty = fields.Float(load_default=0.0)
    frequency_penalty = fields.Float(load_default=0.0)


class ChatCompletionChoiceSchema(Schema):
    """Chat接口消息选择器"""
    index = fields.Int()
    message = fields.Nested(nested=ChatMessageSchema)  # noqa
    finish_reason = fields.Str(
        validate=validate.OneOf(choices=["stop", "length", "content_filter", "function_call"]),  # noqa
        metadata={"example": "stop"})


class ChatCompletionChunkChoiceSchema(Schema):
    """Chat流式消息选择器"""
    index = fields.Int()
    delta = fields.Nested(nested=ChatDeltaSchema)  # noqa
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),  # noqa
        metadata={"example": "stop"})


class ChatCompletionSchema(Schema):
    """Chat接口响应数据结构映射"""
    id = fields.Str(dump_default=lambda: uuid4().hex)
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChoiceSchema))  # noqa
    created = fields.Int(dump_default=lambda: int(time()))
    object = fields.Constant(constant="chat.completion")


class ChatCompletionChunkSchema(Schema):
    """Chat流式响应数据结构映射"""
    id = fields.Str(dump_default=lambda: uuid4().hex)
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChunkChoiceSchema))  # noqa
    created = fields.Int(dump_default=lambda: int(time()))
    object = fields.Constant(constant="chat.completion.chunk")


def create_app() -> Flask:
    """创建接口服务"""
    app = Flask(import_name=__name__)  # 声明主服务
    CORS(app=app)  # 允许跨域
    app.register_blueprint(blueprint=blueprint)  # 注册蓝图
    return app


def get_answer(question: List[Dict[str, str]]) -> str:
    """根据问题文本从接口获取模型回答"""
    url = "{}/run/predict".format(baseURL)
    req = {"fn_index": 0, "data": [[], question[-1]["content"]]}
    resp = post(url=url, data=dumps(req), verify=False).json()
    return resp["data"][0][0][1]


def sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


@stream_with_context
def stream_chat_generate(messages: List[Dict[str, str]]):
    """Chat流式"""
    index = 0
    delta = ChatDeltaSchema().dump({"role": "assistant"})
    choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
    yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    for answer in get_answer(question=messages):
        # 模拟打字机输出效果
        sleep(uniform(0, 0.2))
        index += 1
        delta = ChatDeltaSchema().dump({"content": answer})
        choice = ChatCompletionChunkChoiceSchema().dump({"index": index, "delta": delta, "finish_reason": None})
        yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    index += 1
    choice = ChatCompletionChunkChoiceSchema().dump({"index": index, "delta": {}, "finish_reason": "stop"})
    yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    yield sse(line="[DONE]")


@blueprint.route(rule="/completions", methods=["POST"])
def create_chat_completion() -> str:
    """Chat接口"""
    chat_dict = CreateChatCompletionSchema().load(request.json)
    # 切换到流式
    if chat_dict["stream"]:
        return current_app.response_class(
            response=stream_chat_generate(messages=chat_dict["messages"]), mimetype="text/event-stream")
    answer = get_answer(question=chat_dict["messages"])
    message = ChatMessageSchema().dump({"role": "assistant", "content": answer})
    choice = ChatCompletionChoiceSchema().dump({"index": 0, "message": message, "finish_reason": "stop"})
    return ChatCompletionSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]})  # noqa


if __name__ == "__main__":
    my_app = create_app()  # noqa
    my_app.run(host=appHost, port=appPort, debug=False)
