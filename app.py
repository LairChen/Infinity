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


class ChatRequestSchema(Schema):
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


class ChatChoiceSchema(Schema):
    """Chat流式消息选择器"""
    index = fields.Int(load_default=0)
    delta = fields.Nested(nested=ChatDeltaSchema)  # noqa
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),  # noqa
        metadata={"example": "stop"})


class ChatResponseSchema(Schema):
    """Chat接口响应数据结构映射"""
    id = fields.Str(dump_default=lambda: uuid4().hex)
    created = fields.Int(dump_default=lambda: int(time()))
    model = fields.Str(required=True)  # noqa
    choices = fields.List(fields.Nested(nested=ChatChoiceSchema))  # noqa
    object = fields.Constant(constant="chat.completion.chunk")


def init_model_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """初始化模型和词表"""
    my_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path_eval_finetune,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    my_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=path_eval_finetune,
        use_fast=False,
        trust_remote_code=True
    )
    return my_model, my_tokenizer


def init_frp() -> None:
    """初始化frp客户端"""
    system("chmod +x frpc/frpc")  # noqa
    system("nohup ./frpc/frpc -c frpc/frpc.ini &")  # noqa
    return


def init_api() -> Flask:
    """创建接口服务"""
    my_api = Flask(import_name=__name__)  # 声明主服务
    CORS(app=my_api)  # 允许跨域
    my_api.register_blueprint(blueprint=blueprint)  # 注册蓝图
    return my_api


def init_demo() -> gr.Blocks:
    """创建页面服务"""
    with gr.Blocks(title="Infinity Model") as my_demo:
        # 布局区
        gr.Markdown(value="<p align='center'><img src='https://openi.pcl.ac.cn/rhys2985/Infinity/raw/branch/master/Infinity.png' "
                          "style='height: 100px'/><p>")
        gr.Markdown(value="<center><font size=8>Infinity Chat Bot</center>")
        gr.Markdown(value="<center><font size=4>😸 This Web UI is based on Infinity Model, developed by Rhys. 😸</center>")
        gr.Markdown(value="<center><font size=4>🔥 <a href='https://openi.pcl.ac.cn/rhys2985/Infinity'>项目地址</a> 🔥")
        chatbot = gr.Chatbot(label="Infinity Model")  # noqa
        textbox = gr.Textbox(label="Input", lines=2)
        history = gr.State(value=[])
        with gr.Row():
            btnSubmit = gr.Button("Submit 🚀")
            btnClear = gr.Button("Clear 🧹")
        gr.Markdown(value="<font size=4>⚠ I strongly advise you not to knowingly generate or spread harmful content, "
                          "including rumor, hatred, violence, reactionary, pornography, deception, etc. ⚠")
        # 功能区
        # textbox.submit(fn=chat_with_model, inputs=[chatbot, textbox, history], outputs=[chatbot])
        btnSubmit.click(fn=chat_with_model, inputs=[chatbot, textbox, history], outputs=[chatbot])
        btnSubmit.click(fn=clear_textbox, inputs=[], outputs=[textbox])
        btnClear.click(fn=clear_chatbot_and_history, inputs=[chatbot, history], outputs=[chatbot])
    my_demo.queue()
    return my_demo


def sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


@stream_with_context
def chat_stream(chat_dict: Dict):
    """流式输出模型回答"""
    index = 0
    position = 0
    delta = ChatDeltaSchema().dump({"role": "assistant"})
    choice = ChatChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
    yield sse(line=ChatResponseSchema().dump({"model": chat_dict["model"], "choices": [choice]}))  # noqa
    # 多轮对话，流式输出
    for answer in model.chat(tokenizer, chat_dict["messages"], stream=True):
        if torch.backends.mps.is_available():  # noqa
            torch.mps.empty_cache()  # noqa
        content = answer[position:]
        if not content:
            continue
        delta = ChatDeltaSchema().dump({"content": content})
        choice = ChatChoiceSchema().dump({"index": index, "delta": delta, "finish_reason": None})
        yield sse(line=ChatResponseSchema().dump({"model": chat_dict["model"], "choices": [choice]}))  # noqa
        index += 1
        position = len(answer)
        if position > llm["output_max_length"]:
            break
    choice = ChatChoiceSchema().dump({"index": 0, "delta": {}, "finish_reason": "stop"})
    yield sse(line=ChatResponseSchema().dump({"model": chat_dict["model"], "choices": [choice]}))  # noqa
    yield sse(line="[DONE]")


@blueprint.route(rule="/completions", methods=["POST"])
def chat_completion() -> Response:
    """Chat接口"""
    chat_dict = ChatRequestSchema().load(request.json)
    return current_app.response_class(response=chat_stream(chat_dict=chat_dict), mimetype="text/event-stream")


def chat_with_model(chatbot: List[List[str]], textbox: str, history: List[Dict[str, str]]):  # noqa
    """模型回答并更新聊天窗口"""
    chatbot.append([textbox, ""])
    history.append({"role": "user", "content": textbox})
    # 多轮对话，流式输出
    for answer in model.chat(tokenizer, history, stream=True):
        if torch.backends.mps.is_available():  # noqa
            torch.mps.empty_cache()  # noqa
        chatbot[-1][1] = answer
        yield chatbot
        if len(answer) > llm["output_max_length"]:
            break
    history.append({"role": "assistant", "content": chatbot[-1][1]})


def clear_textbox() -> Dict:
    """清理用户输入空间"""
    return gr.update(value="")


def clear_chatbot_and_history(chatbot: List[List[str]], history: List[Dict[str, str]]) -> List:  # noqa
    """清理人机对话历史记录"""
    chatbot.clear()
    history.clear()
    return chatbot


# 加载模型
model, tokenizer = init_model_and_tokenizer()

# 加载反向代理
init_frp()

# 加载接口服务
api = init_api()  # noqa

# 加载页面服务
demo = init_demo()

# AI协作平台不适用main空间执行，且需要用FastAPI挂载
if __name__ == "__main__":
    # 正式环境启动方法
    api.run(host=appHost, port=appPort, debug=False)
    demo.launch()
else:
    # AI协作平台启动方法
    Thread(target=api.run, kwargs={"host": appHost, "port": appPort, "debug": False}).start()
    app = gr.mount_gradio_app(app=FastAPI(), blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
