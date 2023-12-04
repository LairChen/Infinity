from json import dumps
from os import getenv, system
from threading import Thread
from time import time
from typing import Dict, List, Union
from typing import Tuple
from uuid import uuid4

import gradio as gr
import torch
from fastapi import FastAPI
from flasgger import Schema, fields
from flask import Flask, Blueprint, Response, current_app, request, stream_with_context
from flask_cors import CORS
from marshmallow import validate
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from utils import *

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
    model = fields.Str(required=True)  # noqa
    messages = fields.List(fields.Nested(nested=ChatMessageSchema), required=True)  # noqa
    stream = fields.Bool(load_default=True)
    max_tokens = fields.Int(load_default=None)
    n = fields.Int(load_default=1)
    seed = fields.Int(load_default=1)
    top_p = fields.Float(load_default=1.0)
    temperature = fields.Float(load_default=1.0)
    presence_penalty = fields.Float(load_default=0.0)
    frequency_penalty = fields.Float(load_default=0.0)


class ChatCompletionChunkChoiceSchema(Schema):
    """Chat流式消息选择器"""
    index = fields.Int()
    delta = fields.Nested(nested=ChatDeltaSchema)  # noqa
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),  # noqa
        metadata={"example": "stop"})


class ChatCompletionChunkSchema(Schema):
    """Chat接口响应数据结构映射"""
    id = fields.Str(dump_default=lambda: uuid4().hex)
    created = fields.Int(dump_default=lambda: int(time()))
    model = fields.Str(required=True)  # noqa
    choices = fields.List(fields.Nested(nested=ChatCompletionChunkChoiceSchema))  # noqa
    object = fields.Constant(constant="chat.completion.chunk")


def create_api() -> Flask:
    """创建接口服务"""
    my_api = Flask(import_name=__name__)  # 声明主服务
    CORS(app=my_api)  # 允许跨域
    my_api.register_blueprint(blueprint=blueprint)  # 注册蓝图
    return my_api


def sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


@stream_with_context
def stream_chat_generate(messages: List[Dict[str, str]]):
    """Chat流式"""
    index = 0
    position = 0
    delta = ChatDeltaSchema().dump({"role": "assistant"})
    choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
    yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    for answer in my_model.chat(my_tokenizer, [{"role": "user", "content": messages[-1]["content"]}], stream=True):
        content = answer[position:]
        if torch.backends.mps.is_available():  # noqa
            torch.mps.empty_cache()  # noqa
        if not content:
            continue
        delta = ChatDeltaSchema().dump({"content": content})
        choice = ChatCompletionChunkChoiceSchema().dump({"index": index, "delta": delta, "finish_reason": None})
        yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
        index += 1
        position = len(answer)
    choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": {}, "finish_reason": "stop"})
    yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
    yield sse(line="[DONE]")


@blueprint.route(rule="/completions", methods=["POST"])
def create_chat_completion() -> Response:
    """Chat接口"""
    chat_dict = CreateChatCompletionSchema().load(request.json)
    return current_app.response_class(response=stream_chat_generate(messages=chat_dict["messages"]), mimetype="text/event-stream")


def init_model_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """初始化模型和词表"""
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path_eval_finetune,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=path_eval_finetune,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def chat_with_model(history: List[str], content: str) -> List[Tuple[str, str]]:  # noqa
    """模型回答并更新聊天窗口"""
    response = my_model.chat(my_tokenizer, [{"role": "user", "content": content}])
    if torch.backends.mps.is_available():  # noqa
        torch.mps.empty_cache()  # noqa
    return [(content, response)]


def reset_user_input() -> Dict:
    """清理用户输入空间"""
    return gr.update(value="")


# AI协作平台不适用main空间执行，且需要用FastAPI挂载

# 加载模型
my_model, my_tokenizer = init_model_and_tokenizer()

# 启动frp客户端
system("chmod +x frpc/frpc")  # noqa
system("nohup ./frpc/frpc -c frpc/frpc.ini &")  # noqa

# 接口服务
api = create_api()  # noqa
# 正式环境启动方法
# api.run(host=appHost, port=appPort, debug=False)
# AI协作平台启动方法
Thread(target=api.run, kwargs={"host": appHost, "port": appPort, "debug": False}).start()

# 页面服务
with gr.Blocks(title="Infinity Model") as demo:
    gr.Markdown(value="<p align='center'><img src='https://openi.pcl.ac.cn/rhys2985/Infinity/raw/branch/master/Infinity.png' "
                      "style='height: 100px'/><p>")
    gr.Markdown(value="<center><font size=8>Infinity Chat Bot</center>")
    gr.Markdown(value="<center><font size=4>😸 This Web UI is based on Infinity Model, developed by Rhys. 😸</center>")
    gr.Markdown(value="<center><font size=4>🔥 <a href='https://openi.pcl.ac.cn/rhys2985/Infinity'>项目地址</a> 🔥")
    chatbot = gr.Chatbot(label="Infinity Model")  # noqa
    textbox = gr.Textbox(label="Input", lines=2)
    with gr.Row():
        button = gr.Button("👉 Submit 👈")
    button.click(fn=chat_with_model, inputs=[chatbot, textbox], outputs=[chatbot])
    button.click(fn=reset_user_input, inputs=[], outputs=[textbox])
    gr.Markdown(value="<font size=4>⚠ I strongly advise you not to knowingly generate or spread harmful content, "
                      "including rumor, hatred, violence, reactionary, pornography, deception, etc. ⚠")
# demo.queue()
# 正式环境启动方法
# demo.launch()
# AI协作平台启动方法
app = gr.mount_gradio_app(app=FastAPI(), blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
