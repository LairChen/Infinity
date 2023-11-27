from json import dumps
from time import time
from typing import Dict, List, Union, Tuple
from uuid import uuid4

from flasgger import Schema, fields
from flask import Flask, Blueprint, current_app, request, stream_with_context
from flask_cors import CORS
from marshmallow import validate
from requests import post

baseURL = "https://cloudbrain1notebook.openi.org.cn/notebook_zb29d1e417be4b73b13ed474216bee86_task0"
blueprint = Blueprint(name="Chat", import_name=__name__, url_prefix="/v1/chat")  # 声明蓝图


# 使用marshmallow作序列化和参数校验


class ChatDeltaSchema(Schema):
    role = fields.Str()
    content = fields.Str()


class ChatMessageSchema(Schema):
    role = fields.Str(required=True)
    content = fields.Str(required=True)


class CreateChatCompletionSchema(Schema):
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    messages = fields.List(fields.Nested(nested=ChatMessageSchema), required=True)  # noqa
    max_tokens = fields.Int(load_default=None)
    temperature = fields.Float(load_default=1.0)
    top_p = fields.Float(load_default=1.0)
    n = fields.Int(load_default=1)
    seed = fields.Int(load_default=1)
    stream = fields.Bool(load_default=False)
    presence_penalty = fields.Float(load_default=0.0)
    frequency_penalty = fields.Float(load_default=0.0)


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
    object = fields.Constant(constant="chat.completion.chunk")
    created = fields.Int(dump_default=lambda: int(time()))
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChunkChoiceSchema))  # noqa


class ChatCompletionSchema(Schema):
    id = fields.Str(dump_default=lambda: uuid4().hex)
    object = fields.Constant(constant="chat.completion")
    created = fields.Int(dump_default=lambda: int(time()))
    model = fields.Str(metadata={"example": "baichuan2-7b-chat"})  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChoiceSchema))  # noqa


def create_app() -> Tuple[Flask, Blueprint]:
    """创建接口服务"""
    app = Flask(__name__)  # 声明主服务
    CORS(app=app)  # 允许跨域
    app.register_blueprint(blueprint=blueprint)  # 注册蓝图
    return app, blueprint


def get_answer(question: List) -> str:
    url = "{}/run/predict".format(baseURL)
    req = {"fn_index": 0, "data": [[], question[-1]["content"]]}
    resp = post(url=url, data=dumps(req), verify=False).json()
    return resp["data"][0][0][1]


def sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


@stream_with_context
def stream_chat_generate(messages):
    """Chat流式"""
    delta = ChatDeltaSchema().dump({"role": "assistant"})
    choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
    yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    for answer in get_answer(question=messages):
        delta = ChatDeltaSchema().dump({"content": answer})
        choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
        yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": {}, "finish_reason": "stop"})
    yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    yield sse(line="[DONE]")


@blueprint.route("/completions", methods=["POST"])
def create_chat_completion():
    """Chat接口"""
    chat_dict = CreateChatCompletionSchema().load(request.json)
    # 切换到流式
    if chat_dict["stream"]:
        return current_app.response_class(response=stream_chat_generate(chat_dict["messages"]), mimetype="text/event-stream")
    answer = get_answer(question=chat_dict["messages"])
    message = ChatMessageSchema().dump({"role": "assistant", "content": answer})
    choice = ChatCompletionChoiceSchema().dump({"index": 0, "message": message, "finish_reason": "stop"})
    return ChatCompletionSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]})  # noqa


if __name__ == "__main__":
    my_app, _ = create_app()  # noqa
    my_app.run(host="0.0.0.0", port=8262, debug=False)
