from json import dumps
import time
import uuid
from os import system
from threading import Thread
from typing import Tuple

import torch
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from flasgger import APISpec, Schema, Swagger, fields
from flask import Flask, Blueprint, current_app, request, stream_with_context
from flask_cors import CORS
from marshmallow import validate
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.generation.streamers import TextIteratorStreamer
from transformers.generation.utils import GenerationConfig


class Transformers(object):
    def __init__(self, app=None, tokenizer=None, model=None):
        self.chat = None
        if app is not None:
            self.init_app(app, tokenizer, model)

    def init_app(self, app, tokenizer=None, model=None, chat=None):
        self.tokenizer = tokenizer
        self.model = model
        if chat is None:
            self.chat = model.chat


class ModelSchema(Schema):
    id = fields.Str()
    object = fields.Str(dump_default="model", metadata={"example": "model"})
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    owned_by = fields.Str(dump_default="owner", metadata={"example": "owner"})


class ModelListSchema(Schema):
    object = fields.Str(dump_default="list", metadata={"example": "list"})
    data = fields.List(fields.Nested(ModelSchema), dump_default=[])


class ChatMessageSchema(Schema):
    role = fields.Str(required=True, metadata={"example": "system"})
    content = fields.Str(required=True, metadata={"example": "You are a helpful assistant."})


class CreateChatCompletionSchema(Schema):
    model = fields.Str(required=True, metadata={"example": "gpt-3.5-turbo"})
    messages = fields.List(
        fields.Nested(ChatMessageSchema), required=True,
        metadata={"example": [
            ChatMessageSchema().dump({"role": "system", "content": "You are a helpful assistant."}),
            ChatMessageSchema().dump({"role": "user", "content": "Hello!"})
        ]}
    )
    temperature = fields.Float(load_default=1.0, metadata={"example": 1.0})
    top_p = fields.Float(load_default=1.0, metadata={"example": 1.0})
    n = fields.Int(load_default=1, metadata={"example": 1})
    max_tokens = fields.Int(load_default=None, metadata={"example": None})
    stream = fields.Bool(load_default=False, example=False)
    presence_penalty = fields.Float(load_default=0.0, example=0.0)
    frequency_penalty = fields.Float(load_default=0.0, example=0.0)


class ChatCompletionChoiceSchema(Schema):
    index = fields.Int(metadata={"example": 0})
    message = fields.Nested(ChatMessageSchema, metadata={
        "example": ChatMessageSchema().dump(
            {"role": "assistant", "content": "\n\nHello there, how may I assist you today?"}
        )})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class ChatCompletionSchema(Schema):
    id = fields.Str(
        dump_default=lambda: uuid.uuid4().hex,
        metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("chat.completion")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "gpt-3.5-turbo"})
    choices = fields.List(fields.Nested(ChatCompletionChoiceSchema))


class ChatDeltaSchema(Schema):
    role = fields.Str(metadata={"example": "assistant"})
    content = fields.Str(required=True, metadata={"example": "Hello"})


class ChatCompletionChunkChoiceSchema(Schema):
    index = fields.Int(metadata={"example": 0})
    delta = fields.Nested(ChatDeltaSchema, metadata={"example": ChatDeltaSchema().dump(
        {"role": "assistant", "example": "Hello"})})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class ChatCompletionChunkShema(Schema):
    id = fields.Str(
        dump_default=lambda: uuid.uuid4().hex,
        metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("chat.completion.chunk")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "gpt-3.5-turbo"})
    choices = fields.List(fields.Nested(ChatCompletionChunkChoiceSchema))


class CreateCompletionSchema(Schema):
    model = fields.Str(required=True, metadata={"example": "gpt-3.5-turbo"})
    prompt = fields.Raw(metadata={"example": "Say this is a test"})
    max_tokens = fields.Int(load_default=16, metadata={"example": 256})
    temperature = fields.Float(load_default=1.0, metadata={"example": 1.0})
    top_p = fields.Float(load_default=1.0, metadata={"example": 1.0})
    n = fields.Int(load_default=1, metadata={"example": 1})
    stream = fields.Bool(load_default=False, example=False)
    logit_bias = fields.Dict(load_default=None, example={})
    presence_penalty = fields.Float(load_default=0.0, example=0.0)
    frequency_penalty = fields.Float(load_default=0.0, example=0.0)


class CompletionChoiceSchema(Schema):
    index = fields.Int(load_default=0, metadata={"example": 0})
    text = fields.Str(required=True, metadata={"example": "登鹳雀楼->王之涣\n夜雨寄北->"})
    logprobs = fields.Dict(load_default=None, metadata={"example": {}})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class CompletionUsageSchema(Schema):
    prompt_tokens = fields.Int(metadata={"example": 5})
    completion_tokens = fields.Int(metadata={"example": 7})
    total_tokens = fields.Int(metadata={"example": 12})


class CompletionSchema(Schema):
    id = fields.Str(
        dump_default=lambda: uuid.uuid4().hex,
        metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("text_completion")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "gpt-3.5-turbo"})
    choices = fields.List(fields.Nested(CompletionChoiceSchema))
    usage = fields.Nested(CompletionUsageSchema)


tfs = Transformers()
base_tfs = Transformers()
models_bp = Blueprint(name="Models", import_name=__name__, url_prefix="/v1/models")  # 模型选择
chat_bp = Blueprint(name="Chat", import_name=__name__, url_prefix="/v1/chat")  # Chat模型
completions_bp = Blueprint(name="Completions", import_name=__name__, url_prefix="/v1/completions")  # Completions模型


def sse(line, field="data"):
    return "{}: {}\n\n".format(field, dumps(line, ensure_ascii=False) if isinstance(line, dict) else line)


def empty_cache():
    if torch.backends.mps.is_available():  # noqa
        torch.mps.empty_cache()  # noqa


def init_model() -> Tuple[PeftModelForCausalLM, PreTrainedTokenizerFast]:
    """加载模型和词表"""
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


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(models_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(completions_bp)

    @app.after_request
    def after_request(resp):
        empty_cache()
        return resp

    # Init Swagger
    spec = APISpec(
        title="My OpenAI API",  # noqa
        version="0.0.1",  # noqa
        openapi_version="3.0.2",  # noqa
        plugins=[FlaskPlugin(), MarshmallowPlugin()]  # noqa
    )

    bearer_scheme = {"type": "http", "scheme": "bearer"}
    spec.components.security_scheme("bearer", bearer_scheme)
    template = spec.to_flasgger(app=app, paths=[list_models, create_chat_completion, create_completion])
    app.config["SWAGGER"] = {"openapi": "3.0.2"}  # noqa
    Swagger(app, template=template)

    model, tokenizer = init_model()

    tfs.init_app(app, tokenizer, model)
    base_tfs.init_app(app, tokenizer, model)

    return app


@models_bp.route("")
def list_models():
    model = ModelSchema().dump({"id": "gpt-3.5-turbo"})
    return ModelListSchema().dump({"data": [model]})


@stream_with_context
def stream_chat_generate(messages):
    delta = ChatDeltaSchema().dump(
        {"role": "assistant"})
    choice = ChatCompletionChunkChoiceSchema().dump(
        {"index": 0, "delta": delta, "finish_reason": None})

    yield sse(
        ChatCompletionChunkShema().dump({
            "model": "gpt-3.5-turbo",
            "choices": [choice]})
    )

    position = 0
    for response in tfs.chat(
            tfs.tokenizer,
            messages,
            stream=True):
        content = response[position:]
        if not content:
            continue
        empty_cache()
        delta = ChatDeltaSchema().dump(
            {"content": content})
        choice = ChatCompletionChunkChoiceSchema().dump(
            {"index": 0, "delta": delta, "finish_reason": None})

        yield sse(
            ChatCompletionChunkShema().dump({
                "model": "gpt-3.5-turbo",
                "choices": [choice]})
        )
        position = len(response)

    choice = ChatCompletionChunkChoiceSchema().dump(
        {"index": 0, "delta": {}, "finish_reason": "stop"})

    yield sse(
        ChatCompletionChunkShema().dump({
            "model": "gpt-3.5-turbo",
            "choices": [choice]})
    )

    yield sse("[DONE]")


@chat_bp.route("/completions", methods=["POST"])
def create_chat_completion():
    create_chat_completion = CreateChatCompletionSchema().load(request.json)

    if create_chat_completion["stream"]:
        return current_app.response_class(
            stream_chat_generate(create_chat_completion["messages"]),
            mimetype="text/event-stream"
        )
    else:
        response = tfs.chat(tfs.tokenizer, create_chat_completion["messages"])

        message = ChatMessageSchema().dump(
            {"role": "assistant", "content": response})
        choice = ChatCompletionChoiceSchema().dump(
            {"index": 0, "message": message, "finish_reason": "stop"})
        return ChatCompletionSchema().dump({
            "model": "gpt-3.5-turbo",
            "choices": [choice]})


@stream_with_context
def stream_generate(prompts, **generate_kwargs):
    finish_choices = []
    for index, prompt in enumerate(prompts):
        choice = CompletionChoiceSchema().dump(
            {"index": index, "text": "\n\n", "logprobs": None, "finish_reason": None})

        yield sse(
            CompletionSchema().dump(
                {"model": "gpt-3.5-turbo-instruct", "choices": [choice]})
        )

        inputs = base_tfs.tokenizer(prompt, padding=True, return_tensors="pt")
        inputs = inputs.to(base_tfs.model.device)
        streamer = TextIteratorStreamer(
            base_tfs.tokenizer,
            decode_kwargs={"skip_special_tokens": True})
        Thread(
            target=base_tfs.model.generate, kwargs=dict(
                inputs, streamer=streamer,
                repetition_penalty=1.1, **generate_kwargs)
        ).start()

        finish_reason = None
        for text in streamer:
            if not text:
                continue
            empty_cache()
            if text.endswith(base_tfs.tokenizer.eos_token):
                finish_reason = "stop"
                break

            choice = CompletionChoiceSchema().dump(
                {"index": index, "text": text, "logprobs": None, "finish_reason": None})

            yield sse(
                CompletionSchema().dump(
                    {"model": "gpt-3.5-turbo-instruct", "choices": [choice]})
            )
        else:
            finish_reason = "length"
            choice = CompletionChoiceSchema().dump(
                {"index": index, "text": text, "logprobs": None, "finish_reason": finish_reason})
            yield sse(
                CompletionSchema().dump(
                    {"model": "gpt-3.5-turbo-instruct", "choices": [choice]})
            )

        choice = CompletionChoiceSchema().dump(
            {"index": index, "text": "", "logprobs": None, "finish_reason": finish_reason})
        finish_choices.append(choice)

    yield sse(
        CompletionSchema().dump(
            {"model": "gpt-3.5-turbo-instruct", "choices": finish_choices})
    )

    yield sse("[DONE]")


@completions_bp.route("", methods=["POST"])
def create_completion():
    create_completion = CreateCompletionSchema().load(request.json)

    prompt = create_completion["prompt"]
    prompts = prompt if isinstance(prompt, list) else [prompt]

    if create_completion["stream"]:
        return current_app.response_class(
            stream_generate(prompts, max_new_tokens=create_completion["max_tokens"]),
            mimetype="text/event-stream"
        )
    else:
        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for index, prompt in enumerate(prompts):
            inputs = base_tfs.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(base_tfs.model.device)
            prompt_tokens += len(inputs["input_ids"][0])
            pred = base_tfs.model.generate(
                **inputs, max_new_tokens=create_completion["max_tokens"], repetition_penalty=1.1)

            completion_tokens += len(pred.cpu()[0])
            resp = base_tfs.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

            finish_reason = None
            if resp.endswith(base_tfs.tokenizer.eos_token):
                finish_reason = "stop"
                resp = resp[:-len(base_tfs.tokenizer.eos_token)]
            else:
                finish_reason = "length"

            choices.append(
                CompletionChoiceSchema().dump(
                    {"index": index, "text": resp, "logprobs": {}, "finish_reason": finish_reason})
            )
        usage = CompletionUsageSchema().dump({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens})

        return CompletionSchema().dump(
            {"model": "gpt-3.5-turbo-instruct", "choices": choices, "usage": usage})


if __name__ == "__main__":
    system("chmod +x frpc/frpc")  # noqa
    system("nohup ./frpc/frpc -c frpc/frpc.ini &")  # noqa
    myapp = create_app()  # noqa
    myapp.run(host="0.0.0.0", port=8262, debug=False)
