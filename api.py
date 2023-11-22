import os

import gradio as gr
from fastapi import FastAPI

app = FastAPI()

demo = gr.Interface(
    fn=lambda x: x,
    inputs=gr.components.Textbox(label='Input'),
    outputs=gr.components.Textbox(label='Output'),
    allow_flagging='never'
)

app = gr.mount_gradio_app(app, demo, path=os.getenv('OPENI_GRADIO_URL'))
