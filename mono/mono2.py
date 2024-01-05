from base64 import b64decode
from json import loads
from os import environ, getenv, system
from threading import Thread
from typing import List

import gradio as gr
from cv2 import imread
from fastapi import FastAPI
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from paddlehub import Module

# online inference only for pyramid box

# 全局模型
model = None
environ["CUDA_VISIBLE_DEVICES"] = "0"  # noqa


def detect(path: str) -> List[List[int]]:
    """读取指定路径的图片文件，检测人脸并返回矩形坐标"""
    global model
    ans = []
    if model is not None:
        result = model.face_detection(images=[imread(filename=path)], use_gpu=True)  # noqa
        for loc in result[0]["data"]:
            # 按照上下左右的顺序进行组织
            ans.append([loc["top"], loc["bottom"], loc["left"], loc["right"]])
    return ans


def init_model() -> None:
    """加载Pyramid Box模型"""
    global model
    model = Module(name="pyramidbox_face_detection")  # noqa
    return


def init_api() -> Flask:
    """创建接口服务"""
    system("chmod +x frpc/frpc")  # noqa
    system("nohup ./frpc/frpc -c frpc/frpc.ini &")  # noqa
    my_api = Flask(import_name=__name__)  # 声明主服务
    CORS(app=my_api)  # 允许跨域
    return my_api


api = init_api()


@api.route(rule="/", methods=["GET"])
def index() -> str:
    """接口服务首页"""
    return render_template(template_name_or_list="Infinity.html")  # noqa


@api.route(rule="/detect", methods=["POST"])
def detect_api() -> Response:
    """接口服务专用"""
    b64 = loads(s=request.json)["b64"]
    print(b64decode(s=b64))
    return jsonify([])


def detect_gr() -> str:
    """页面服务专用"""
    pass


def init_demo() -> gr.Blocks:
    """创建Image Bot主页面"""
    with gr.Blocks(title="Infinity Model") as my_demo:
        return my_demo


Thread(target=init_model).start()
Thread(target=api.run, kwargs={"host": "0.0.0.0", "port": 8262, "debug": False}).start()
demo = init_demo()
# 正式环境启动方法
if __name__ == "__main__":
    demo.launch()
# AI协作平台启动方法
else:
    app = gr.mount_gradio_app(app=FastAPI(), blocks=demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
