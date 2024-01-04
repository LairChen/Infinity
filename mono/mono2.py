from typing import List

from cv2 import imread
from paddlehub import Module

# online inference only for pyramid box

# 全局模型
model = None


def detect(path: str) -> List[List[int]]:
    model.face_detection(  # noqa
        images=[imread(filename=path)]
    )


def init_model() -> None:
    """加载Pyramid Box模型"""
    global model
    model = Module(name="pyramidbox_face_detection")  # noqa
    return
