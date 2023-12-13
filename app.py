from json import dumps
from os import getenv, listdir, system
from threading import Thread
from typing import Dict, List, Union, Tuple, Optional

import gradio as gr
import numpy as np
import torch
from fastapi import FastAPI
from flask import Flask, Response, request, current_app, render_template, stream_with_context
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from tiktoken import get_encoding
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *


# STEP1.加载模型
# 包括对话模型和嵌入模型，其中对话模型从model获取，嵌入模型从dataset获取
# 对话模型必须，嵌入模型可选
# model模型类别从model_type.txt文件中获取，dataset模型类别从压缩文件名获取


def init_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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


def init_embeddings_model() -> Optional[SentenceTransformer]:
    """初始化嵌入模型"""
    for filename in listdir(path_eval_pretrain):
        modelname = match(pattern="(.*)\.zip", string=filename)  # noqa
        if modelname is not None:
            my_model_name = modelname.groups()[0]
            break
    else:
        return None
    system("unzip {}/{}.zip -d {}".format(path_eval_pretrain, my_model_name, path_eval_pretrain))
    my_model = SentenceTransformer(
        model_name_or_path="/dataset/m3e-large",
        device="cuda"  # noqa
    )
    return my_model


model, tokenizer = init_model_and_tokenizer()
embeddings_model = init_embeddings_model()


# STEP2.启动接口服务
# 接口服务均采用线程启动，避免阻塞
# 采用frp工具反向代理


def init_api() -> Flask:
    """创建接口服务"""
    my_api = Flask(import_name=__name__)  # 声明主服务
    CORS(app=my_api)  # 允许跨域
    return my_api


def init_frp() -> None:
    """初始化frp客户端"""
    system("chmod +x frpc/frpc")  # noqa
    system("nohup ./frpc/frpc -c frpc/frpc.ini &")  # noqa
    return


api = init_api()
Thread(target=api.run, kwargs={"host": appHost, "port": appPort, "debug": False}).start()
init_frp()


@api.route(rule="/", methods=["GET"])
def homepage():
    """接口服务首页"""
    return render_template(template_name_or_list="Infinity.html")  # noqa


@api.route(rule="/v1/chat/completions", methods=["POST"])
def chat_completions() -> Response:
    """Chat接口"""
    req = ChatRequestSchema().load(request.json)
    return current_app.response_class(response=chat_stream(req=req), mimetype="text/event-stream")


@api.route(rule="/v1/embeddings", methods=["POST"])
def embeddings() -> str:
    """Embeddings接口"""
    if embeddings_model is None:
        return ""
    req = EmbeddingsRequestSchema().load(request.json)
    result = [embeddings_model.encode(sentences=sentence) for sentence in req["input"]]
    # OpenAI API 嵌入维度标准1536
    result = [embeddings_pad(embedding=embedding, target_length=embeddingLength)
              if len(embedding) < embeddingLength else embedding for embedding in result]
    result = [embedding / np.linalg.norm(x=embedding) for embedding in result]
    result = [embedding.tolist() for embedding in result]
    prompt_tokens = sum(len(text.split()) for text in req["input"])
    total_tokens = sum(embeddings_token_num(text=text) for text in req["input"])
    data = [{"index": index, "embedding": embedding} for index, embedding in enumerate(result)]
    usage = {"prompt_tokens": prompt_tokens, "total_tokens": total_tokens}
    return EmbeddingsResponseSchema().dump({"model": req["model"], "data": data, "usage": usage})


@stream_with_context
def chat_stream(req: Dict):
    """流式输出模型回答"""
    index = 0
    position = 0
    delta = ChatDeltaSchema().dump({"role": "assistant"})
    choice = ChatChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
    yield chat_sse(line=ChatResponseSchema().dump({"model": req["model"], "choices": [choice]}))  # noqa
    # 多轮对话，流式输出
    # 接口使用字符式
    for answer in model.chat(tokenizer, req["messages"], stream=True):  # noqa
        if torch.backends.mps.is_available():  # noqa
            torch.mps.empty_cache()  # noqa
        content = answer[position:]
        if not content:
            continue
        delta = ChatDeltaSchema().dump({"content": content})
        choice = ChatChoiceSchema().dump({"index": index, "delta": delta, "finish_reason": None})
        yield chat_sse(line=ChatResponseSchema().dump({"model": req["model"], "choices": [choice]}))  # noqa
        index += 1
        position = len(answer)
        if position > contentLength:
            break
    choice = ChatChoiceSchema().dump({"index": 0, "delta": {}, "finish_reason": "stop"})
    yield chat_sse(line=ChatResponseSchema().dump({"model": req["model"], "choices": [choice]}))  # noqa
    yield chat_sse(line="[DONE]")


def chat_sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


def embeddings_pad(embedding: np.ndarray, target_length: int) -> np.ndarray:
    """按照指定维度对嵌入向量进行扩缩"""
    embedding = PolynomialFeatures(degree=2).fit_transform(X=embedding.reshape(1, -1))
    embedding = embedding.flatten()
    # 维度小填充，维度大截断
    if len(embedding) < target_length:
        return np.pad(array=embedding, pad_width=(0, target_length - len(embedding)))
    return embedding[:target_length]


def embeddings_token_num(text: str) -> int:
    """计算嵌入消耗"""
    return len(get_encoding(encoding_name="cl100k_base").encode(text=text))


# STEP3.启动页面服务
# AI协作平台不适用main空间执行，且需要用FastAPI挂载


def get_answer(chatbot: List[List[str]], textbox: str, history: List[Dict[str, str]]):  # noqa
    """模型回答并更新聊天窗口"""
    chatbot.append([textbox, ""])
    history.append({"role": "user", "content": textbox})
    # 多轮对话，流式输出
    # 页面使用段落式
    for answer in model.chat(tokenizer, history, stream=True):  # noqa
        if torch.backends.mps.is_available():  # noqa
            torch.mps.empty_cache()  # noqa
        chatbot[-1][1] = answer
        yield chatbot
        if len(answer) > contentLength:
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


def init_demo() -> gr.Blocks:
    """创建页面服务"""
    with gr.Blocks(title="Infinity Model") as my_demo:
        # 布局区
        gr.Markdown(value="<p align='center'>"
                          "<img src='https://openi.pcl.ac.cn/rhys2985/Infinity/raw/branch/master/templates/Infinity.png' "
                          "style='height: 100px'>"
                          "</p>")
        gr.Markdown(value="<center><font size=8>Infinity Chat Bot</center>")
        gr.Markdown(value="<center><font size=4>😸 This Web UI is based on Infinity Model, developed by Rhys. 😸</center>")
        gr.Markdown(value="<center><font size=4>🔥 <a href='https://openi.pcl.ac.cn/rhys2985/Infinity'>项目地址</a> 🔥</center>")
        chatbot = gr.Chatbot(label="Infinity Model")  # noqa
        textbox = gr.Textbox(label="Input", lines=2)
        history = gr.State(value=[])
        with gr.Row():
            btnSubmit = gr.Button("Submit 🚀")
            btnClear = gr.Button("Clear 🧹")
        gr.Markdown(value="<center><font size=4>⚠ I strongly advise you not to knowingly generate or spread harmful content, "
                          "including rumor, hatred, violence, reactionary, pornography, deception, etc. ⚠</center>")
        # 功能区
        btnSubmit.click(fn=get_answer, inputs=[chatbot, textbox, history], outputs=[chatbot])
        btnSubmit.click(fn=clear_textbox, inputs=[], outputs=[textbox])
        btnClear.click(fn=clear_chatbot_and_history, inputs=[chatbot, history], outputs=[chatbot])
    my_demo.queue()
    return my_demo


demo = init_demo()
# 正式环境启动方法
if __name__ == "__main__":
    demo.launch()
# AI协作平台启动方法
else:
    app = gr.mount_gradio_app(app=FastAPI(), blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
