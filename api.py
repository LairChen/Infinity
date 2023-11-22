import os

import gradio as gr
from fastapi import FastAPI

app = FastAPI()
def create_chat_completion():
    """Chat接口"""
    return 213
demo = gr.Interface(
    fn=create_chat_completion,
    inputs=gr.components.Textbox(label='Input'),
    outputs=gr.components.Textbox(label='Output'),
    allow_flagging='never'
)
os.system("mkdir /tmp/dataset")
os.system("unzip /dataset/Baichuan2-7B-Chat.zip -d /tmp/dataset")
app = gr.mount_gradio_app(app, demo, path=os.getenv('OPENI_GRADIO_URL'))
