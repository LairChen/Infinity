from json import dumps
from os import getenv
from typing import Union, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from tiktoken import get_encoding
from uvicorn import run

from utils import *

# 包括对话模型/补全模型和嵌入模型，其中对话模型/补全模型从pretrainmodel获取，嵌入模型从dataset获取
# 对话模型/补全模型必选，嵌入模型可选
# pretrainmodel模型类别从model_type.txt文件中获取，dataset模型类别从压缩文件名获取

language_model = init_language_model()
embedding_model = init_embedding_model()

# 初始化接口服务，指定路由前缀
# AI协作平台上从环境变量中获取路由前缀，其他情况下忽略

prefix = getenv(key="OPENI_GRADIO_URL", default="")  # noqa
app = FastAPI()


def sse(line: Union[str, Dict]) -> str:
    """Server Sent Events for stream"""
    return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


@app.get(path=prefix + "/", response_class=HTMLResponse)
def homepage():
    """接口服务首页"""
    return open(file="templates/Infinity.html", mode="r", encoding="utf-8").read()


@app.post(path=prefix + "/v1/chat/completions", response_model=None)
def chat(args: Dict) -> Union[StreamingResponse, Dict]:
    """Chat接口"""
    req = ChatRequestSchema().load(args)
    if req["stream"]:
        # 流式响应
        return StreamingResponse(content=chat_stream(req=req), media_type="text/event-stream")
    else:
        # 非流式响应
        return chat_generate(req=req)


# @app.post(path=prefix + "/v1/completions", response_model=None)
# def completions(args: Dict) -> Union[StreamingResponse, Dict]:
#     """Completions接口"""
#     req = CompletionsRequestSchema().load(args)
#     if req["stream"]:
#         # 流式响应
#         pass
#     else:
#         # 非流式响应
#         pass


@app.post(path=prefix + "/v1/embeddings", response_class=JSONResponse)
def embeddings(args: Dict) -> Dict:
    """Embeddings接口"""
    req = EmbeddingsRequestSchema().load(args)
    return embeddings_result(req=req)


def chat_generate(req: Dict) -> Dict:
    """输出模型回答"""
    message = ChatMessageSchema().dump({"role": "assistant", "content": language_model.generate(conversation=req["messages"])})
    choice = ChatChoiceSchema().dump({"index": 0, "message": message})
    return ChatResponseSchema().dump({"model": language_model.name, "choices": [choice]})


def chat_stream(req: Dict):
    """流式输出模型回答"""
    index = 0
    delta = ChatMessageSchema().dump({"role": "assistant", "content": ""})
    choice = ChatChoiceChunkSchema().dump({"index": index, "delta": delta, "finish_reason": None})
    yield sse(line=ChatResponseChunkSchema().dump({"model": language_model.name, "choices": [choice]}))
    # 多轮对话，字符型流式输出
    for answer in language_model.stream(conversation=req["messages"]):
        index += 1
        delta = ChatMessageSchema().dump({"role": "assistant", "content": answer})
        choice = ChatChoiceChunkSchema().dump({"index": index, "delta": delta, "finish_reason": None})
        yield sse(line=ChatResponseChunkSchema().dump({"model": language_model.name, "choices": [choice]}))
    choice = ChatChoiceChunkSchema().dump({"index": 0, "delta": {}, "finish_reason": "stop"})
    yield sse(line=ChatResponseChunkSchema().dump({"model": language_model.name, "choices": [choice]}))
    yield sse(line="[DONE]")


def embeddings_result(req: Dict) -> Dict:
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


# AI协作平台不适用main空间执行，需要返回FastAPI对象
if __name__ == "__main__":
    run(app=app, host=appHost, port=appPort)
