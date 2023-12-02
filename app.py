from os import getenv
from typing import List, Tuple, Dict

import gradio as gr
import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def init_model_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """初始化模型和词表"""
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="/pretrainmodel",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/pretrainmodel",
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


# AI协作平台自有FastAPI服务，这里模块式运行Gradio服务并挂载，故不适用main空间执行
my_model, my_tokenizer = init_model_and_tokenizer()
app = FastAPI()
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
app = gr.mount_gradio_app(app=app, blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
