from os import getenv

import gradio as gr
from fastapi import FastAPI


# online inference only for stable diffusion


def paint(prompt: str, artist: str, style: str, width: int, height: int) -> str:
    """收集用户收入并完成绘画"""
    return "https://openi.pcl.ac.cn/rhys2985/Infinity/raw/branch/master/templates/Infinity.png"


def webui() -> gr.Blocks:
    """创建Image Bot主页面"""
    with gr.Blocks(title="Infinity Model") as my_demo:
        # 布局区
        gr.Markdown(value="<p align='center'>"
                          "<img src='https://openi.pcl.ac.cn/rhys2985/Infinity/raw/branch/master/templates/Infinity.png' "
                          "style='height: 100px'>"
                          "</p>")
        gr.Markdown(value="<center><font size=8>Infinity Image Bot</center>")
        gr.Markdown(value="<center><font size=4>😸 This Web UI is based on Infinity Model, developed by Rhys. 😸</center>")
        gr.Markdown(value="<center><font size=4>🔥 <a href='https://openi.pcl.ac.cn/rhys2985/Infinity'>项目地址</a> 🔥</center>")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your circumstantial prompt", lines=3)
                artist = gr.Textbox(label="Artist", placeholder="Choose your favourite artist")
                style = gr.Textbox(label="Style", placeholder="Appoint your painting style")
                width = gr.Slider(label="Width(64x)", minimum=64, maximum=1024, value=512, step=64)
                height = gr.Slider(label="Height(64x)", minimum=64, maximum=1024, value=512, step=64)
                submit = gr.Button("Submit 🚀")
            display = gr.Image(label="Infinity Model")
        gr.Markdown(value="<center><font size=4>⚠ I strongly advise you not to knowingly generate or spread harmful content, "
                          "including rumor, hatred, violence, reactionary, pornography, deception, etc. ⚠</center>")
        # 功能区
        submit.click(fn=paint, inputs=[prompt, artist, style, width, height], outputs=[display])
        return my_demo


demo = webui()
# 正式环境启动方法
if __name__ == "__main__":
    demo.launch()
# AI协作平台启动方法
else:
    app = gr.mount_gradio_app(app=FastAPI(), blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
