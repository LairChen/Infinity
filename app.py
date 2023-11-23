from os import system, getenv
from typing import List, Tuple

import gradio as gr
import torch
from fastapi import FastAPI
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig


def init_env() -> None:
    system("mkdir /tmp/dataset")
    system("unzip /pretrainmodel/Baichuan2-7B-Chat.zip -d /tmp/dataset")
    return


def init_model():
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


# def sse(line: Union[str, Dict]) -> str:
#     """Server Sent Events for stream"""
#     return "data: {}\n\n".format(dumps(obj=line, ensure_ascii=False) if isinstance(line, dict) else line)


# @stream_with_context
# def stream_chat_generate(messages):
#     """Chat流式"""
#     delta = ChatDeltaSchema().dump({"role": "assistant"})
#     choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
#     yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
#     position = 0
#     for response in my_model.chat(my_tokenizer, messages, stream=True):
#         content = response[position:]
#         if not content:
#             continue
#         if torch.backends.mps.is_available():  # noqa
#             torch.mps.empty_cache()  # noqa
#         delta = ChatDeltaSchema().dump({"content": content})
#         choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": delta, "finish_reason": None})
#         yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
#         position = len(response)
#     choice = ChatCompletionChunkChoiceSchema().dump({"index": 0, "delta": {}, "finish_reason": "stop"})
#     yield sse(line=ChatCompletionChunkSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]}))  # noqa
#     yield sse(line="[DONE]")


def chat_with_model(content: str) -> List[Tuple[str]]:
    """Chat接口"""
    result = my_model.chat(my_tokenizer, [{"role": "user", "content": content}])
    if torch.backends.mps.is_available():  # noqa
        torch.mps.empty_cache()  # noqa
    return [(content, result)]
    # chat_dict = CreateChatCompletionSchema().load(request.json)
    # # if chat_dict["stream"]:
    # #     # 切换到流式
    # #     return current_app.response_class(response=stream_chat_generate(chat_dict["messages"]), mimetype="text/event-stream")
    # response = my_model.chat(my_tokenizer, chat_dict["messages"])
    # message = ChatMessageSchema().dump({"role": "assistant", "content": response})
    # choice = ChatCompletionChoiceSchema().dump({"index": 0, "message": message, "finish_reason": "stop"})
    # return ChatCompletionSchema().dump({"model": "baichuan2-7b-chat", "choices": [choice]})  # noqa


# AI协作平台自有FastAPI服务，这里模块式运行Gradio服务并挂载，故不适用main空间执行
init_env()
my_model, my_tokenizer = init_model()
app = FastAPI()


def reset_user_input():
    return gr.update(value="")


with gr.Blocks() as demo:
    gr.Markdown(value="<p align='center'><img src='https://openi.pcl.ac.cn/rhys2985/Infinity-llm/raw/branch/master/infinity.png' "
                      "style='height: 100px'/><p>")
    gr.Markdown(value="<center><font size=8>Infinity Chat Bot</center>")
    gr.Markdown(value="<center><font size=4>😸 This Web UI is based on Infinity Model, developed by Rhys. 😸</center>")
    gr.Markdown(value="<center><font size=4>🔥 <a href='https://openi.pcl.ac.cn/rhys2985/Infinity-llm'>项目地址</a> 🔥")

    chatbot = gr.Chatbot(label="Infinity Model", elem_classes="control-height")  # noqa
    query = gr.Textbox(lines=2, label="Input")
    task_history = gr.State([])

    with gr.Row():
        submit_btn = gr.Button("Submit 🚀")

    submit_btn.click(chat_with_model, [query, chatbot, task_history], [chatbot])
    submit_btn.click(reset_user_input, [], [query])

    gr.Markdown(value="<font size=4>⚠ I strongly advise you not to knowingly generate or spread harmful content, "
                      "including rumor, hatred, violence, reactionary, pornography, deception, etc. ⚠")
app = gr.mount_gradio_app(app, demo, path=getenv("OPENI_GRADIO_URL"))  # noqa
