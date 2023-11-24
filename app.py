from os import system, getenv
from typing import Tuple, Dict

import gradio as gr
import torch
from fastapi import FastAPI
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.generation.utils import GenerationConfig


def init_env() -> None:
    """环境初始化"""
    system("mkdir /tmp/dataset")
    system("unzip /pretrainmodel/Baichuan2-7B-Chat.zip -d /tmp/dataset")
    return


def init_model() -> Tuple[PeftModelForCausalLM, PreTrainedTokenizer]:
    """模型和词表初始化"""
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


def chat_with_model(content: str):  # noqa
    """模型流式输出"""
    for response in my_model.chat(my_tokenizer, [{"role": "user", "content": content}], stream=True):
        if torch.backends.mps.is_available():  # noqa
            torch.mps.empty_cache()  # noqa
        yield [(content, response)]


def reset_user_input() -> Dict:
    """清理用户输入空间"""
    return gr.update(value="")


# AI协作平台自有FastAPI服务，这里模块式运行Gradio服务并挂载，故不适用main空间执行
init_env()
my_model, my_tokenizer = init_model()
app = FastAPI()
demo = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(
        label="Ask a question", placeholder="What is the capital of France?"
    ),
    outputs=[gr.Textbox(label="Answer"), gr.Number(label="Score")],
    allow_flagging="never"
)
# with gr.Blocks(title="Infinity Model") as demo:
#     gr.Markdown(value="<p align='center'><img src='https://openi.pcl.ac.cn/rhys2985/Infinity-llm/raw/branch/master/infinity.png' "
#                       "style='height: 100px'/><p>")
#     gr.Markdown(value="<center><font size=8>Infinity Chat Bot</center>")
#     gr.Markdown(value="<center><font size=4>😸 This Web UI is based on Infinity Model, developed by Rhys. 😸</center>")
#     gr.Markdown(value="<center><font size=4>🔥 <a href='https://openi.pcl.ac.cn/rhys2985/Infinity-llm'>项目地址</a> 🔥")
#     chatbot = gr.Chatbot(label="Infinity Model", elem_classes="control-height")  # noqa
#     textbox = gr.Textbox(lines=2, label="Input")
#     with gr.Row():
#         submit_btn = gr.Button("👉 Submit 👈")
#     submit_btn.click(chat_with_model, [chatbot, textbox], [chatbot])
#     submit_btn.click(reset_user_input, [], [textbox])
#     gr.Markdown(value="<font size=4>⚠ I strongly advise you not to knowingly generate or spread harmful content, "
#                       "including rumor, hatred, violence, reactionary, pornography, deception, etc. ⚠")
demo.queue()
app = gr.mount_gradio_app(app, demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
