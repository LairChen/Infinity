from os import system, getenv
from threading import Thread

import gradio as gr
from fastapi import FastAPI
from flask import Flask

myapp = Flask(__name__)


@myapp.route('/')
def hello_flask():
    return 'Hello Flask!'


system("chmod +x frpc/frpc")
system("nohup ./frpc/frpc -c frpc/frpc.ini &")

Thread(target=myapp.run, kwargs={"port": 8999}).start()
app = FastAPI()
with gr.Blocks(title="Infinity Model") as demo:
    gr.Markdown(value="<p align='center'><img src='https://openi.pcl.ac.cn/rhys2985/Infinity/raw/branch/master/Infinity.png' "
                      "style='height: 100px'/><p>")
app = gr.mount_gradio_app(app=app, blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
