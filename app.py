from os import getenv
from typing import Union, Tuple, List, Dict

import gradio as gr
from fastapi import FastAPI

from utils import *


def init_language_model() -> Union[ChatModel, CompletionModel]:
    """初始化模型和词表"""
    with open(file="{}/model_type.txt".format(path_eval_finetune), mode="r", encoding="utf-8") as f:
        my_model_name = f.read().strip()
    if CHAT_MODEL_TYPE.get(my_model_name, None) is not None:
        my_model = CHAT_MODEL_TYPE[my_model_name](name=my_model_name, path=path_eval_finetune)
    elif COMPLETION_MODEL_TYPE.get(my_model_name, None) is not None:
        my_model = COMPLETION_MODEL_TYPE[my_model_name](name=my_model_name, path=path_eval_finetune)
    else:
        raise FileNotFoundError("no existing language model")
    return my_model


model = init_language_model()


def submit(chatbot: List[List[str]], textbox: str, history: List[Dict[str, str]]) -> Tuple[List[List[str]], str]:  # noqa
    """模型回答并更新聊天窗口"""
    history.append({"role": "user", "content": textbox})
    answer = model.generate(conversation=history)  # 多轮对话，非流式文本输出
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
        gr.Markdown(value="<center><font size=8>Infinity Large Language Model</center>")
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


# AI协作平台不适用main空间执行，需要用FastAPI挂载
demo = init_demo()
# 正式环境启动方法
if __name__ == "__main__":
    demo.launch()
# AI协作平台启动方法
else:
    app = gr.mount_gradio_app(app=FastAPI(), blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
