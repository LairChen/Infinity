from json import dumps
from os import getenv, listdir
from re import match
from typing import Union, Optional, Tuple

import gradio as gr
from fastapi import FastAPI
from flask import Flask, Response, current_app, jsonify, render_template, request, stream_with_context
from flask_cors import CORS
from tiktoken import get_encoding

from utils import *


# STEP1.加载模型
# 包括对话模型/补全模型和嵌入模型，其中对话模型/补全模型从pretrainmodel获取，嵌入模型从dataset获取
# 对话模型/补全模型必选，嵌入模型可选
# pretrainmodel模型类别从model_type.txt文件中获取，dataset模型类别从压缩文件名获取


def init_language_model() -> Union[BaseChatModel, BaseCompletionModel]:
    """初始化模型和词表"""
    with open(file="{}/model_type.txt".format(path_eval_finetune), mode="r", encoding="utf-8") as f:
        my_model_name = f.read()
    if CHAT_MODEL_TYPE.get(my_model_name, None) is not None:
        my_model = CHAT_MODEL_TYPE[my_model_name](name=my_model_name, path=path_eval_finetune)
    elif COMPLETION_MODEL_TYPE.get(my_model_name, None) is not None:
        my_model = COMPLETION_MODEL_TYPE[my_model_name](name=my_model_name, path=path_eval_finetune)
    else:
        raise FileNotFoundError("no existing language model")
    return my_model


def init_embedding_model() -> Optional[BaseEmbeddingModel]:
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


language_model = init_language_model()
embedding_model = init_embedding_model()


# STEP2.启动接口服务
# 接口服务均采用线程启动，避免阻塞
# 采用frp工具反向代理


def init_api() -> Flask:
    """创建接口服务"""
    my_api = Flask(import_name=__name__)  # 声明主服务
    CORS(app=my_api)  # 允许跨域
    return my_api


api = init_api()
Thread(target=api.run, kwargs={"host": appHost, "port": appPort, "debug": False}).start()


@api.route(rule="/", methods=["GET"])
def homepage() -> str:
    """接口服务首页"""
    return render_template(template_name_or_list="Infinity.html")  # noqa


@api.route(rule="/v1/chat/completions", methods=["POST"])
def chat() -> Response:
    """Chat接口"""
    req = ChatRequestSchema().load(request.json)
    if req["stream"]:
        # 流式响应
        return current_app.response_class(response=chat_stream(req=req), mimetype="text/event-stream")
    else:
        # 非流式响应
        return jsonify(chat_result(req=req))


# @api.route(rule="/v1/completions", methods=["POST"])
# def completions() -> Response:
#     """Completions接口"""
#     return jsonify("")


@api.route(rule="/v1/embeddings", methods=["POST"])
def embeddings() -> Response:
    """Embeddings接口"""
    req = EmbeddingsRequestSchema().load(request.json)
    return jsonify(embeddings_result(req=req))


def chat_result(req: Dict) -> str:
    """输出模型回答"""
    message = ChatMessageSchema().dump({"role": "assistant", "content": language_model.generate(conversation=req["messages"])})
    choice = ChatChoiceSchema().dump({"index": 0, "message": message})
    return ChatResponseSchema().dump({"model": language_model.name, "choices": [choice]})


@stream_with_context
def chat_stream(req: Dict):
    """流式输出模型回答"""
    index = 0
    delta = ChatMessageSchema().dump({"role": "assistant", "content": ""})
    choice = ChatChoiceChunkSchema().dump({"index": index, "delta": delta, "finish_reason": None})
    yield chat_sse(line=ChatResponseChunkSchema().dump({"model": language_model.name, "choices": [choice]}))
    # 多轮对话，字符型流式输出
    for answer in language_model.stream(conversation=req["messages"]):
        index += 1
        delta = ChatMessageSchema().dump({"role": "assistant", "content": answer})
        choice = ChatChoiceChunkSchema().dump({"index": index, "delta": delta, "finish_reason": None})
        yield chat_sse(line=ChatResponseChunkSchema().dump({"model": language_model.name, "choices": [choice]}))
    choice = ChatChoiceChunkSchema().dump({"index": 0, "delta": {}, "finish_reason": "stop"})
    yield chat_sse(line=ChatResponseChunkSchema().dump({"model": language_model.name, "choices": [choice]}))
    yield chat_sse(line="[DONE]")


def chat_sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


def embeddings_result(req: Dict) -> str:
    """计算嵌入结果"""
    data = [{"index": index, "embedding": embedding_model.embedding(sentence=text) if embedding_model is not None else []}
            for index, text in enumerate(req["input"])]
    usage = {
        "prompt_tokens": sum(len(text.split()) for text in req["input"]),
        "total_tokens": sum(embeddings_token_num(text=text) for text in req["input"])
    }
    return EmbeddingsResponseSchema().dump({
        "model": embedding_model.name if embedding_model is not None else "", "data": data, "usage": usage})


def embeddings_token_num(text: str) -> int:
    """计算嵌入消耗"""
    return len(get_encoding(encoding_name="cl100k_base").encode(text=text))


# STEP3.启动页面服务
# AI协作平台不适用main空间执行，且需要用FastAPI挂载


def submit(chatbot: List[List[str]], textbox: str, history: List[Dict[str, str]]) -> Tuple[List[List[str]], str]:  # noqa
    """模型回答并更新聊天窗口"""
    history.append({"role": "user", "content": textbox})
    answer = language_model.generate(conversation=history)  # 多轮对话，非流式文本输出
    history.append({"role": "assistant", "content": answer})
    chatbot.append([textbox, answer])
    return chatbot, ""


def clean(chatbot: List[List[str]], history: List[Dict[str, str]]) -> List[List[str]]:  # noqa
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
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="Infinity Model")  # noqa
                textbox = gr.Textbox(label="Input", lines=2)
                history = gr.State(value=[])
                with gr.Row():
                    btnSubmit = gr.Button("Submit 🚀")
                    btnClean = gr.Button("Clean 🧹")
            with gr.Column(scale=1):
                pass
        gr.Markdown(value="<center><font size=4>⚠ I strongly advise you not to knowingly generate or spread harmful content, "
                          "including rumor, hatred, violence, reactionary, pornography, deception, etc. ⚠</center>")
        # 功能区
        btnSubmit.click(fn=submit, inputs=[chatbot, textbox, history], outputs=[chatbot, textbox])
        btnClean.click(fn=clean, inputs=[chatbot, history], outputs=[chatbot])
    return my_demo


demo = init_demo()
# 正式环境启动方法
if __name__ == "__main__":
    demo.launch()
# AI协作平台启动方法
else:
    app = gr.mount_gradio_app(app=FastAPI(), blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
