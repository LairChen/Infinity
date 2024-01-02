from time import time
from uuid import uuid4

from flasgger import Schema, fields
from marshmallow import validate


# 使用marshmallow做序列化和参数校验

# Chat接口

class ChatMessageSchema(Schema):
    """Chat消息结构映射"""
    role = fields.Str(required=True)
    content = fields.Str(required=True)


class ChatRequestSchema(Schema):
    """Chat接口请求数据结构解析"""
    model = fields.Str(required=True)
    messages = fields.List(fields.Nested(nested=ChatMessageSchema), required=True)  # noqa
    stream = fields.Bool(required=True)
    max_tokens = fields.Int(load_default=1024)
    n = fields.Int(load_default=1)
    seed = fields.Int(load_default=1)
    top_p = fields.Float(load_default=1.0)
    temperature = fields.Float(load_default=1.0)
    presence_penalty = fields.Float(load_default=0.0)
    frequency_penalty = fields.Float(load_default=0.0)


class ChatChoiceSchema(Schema):
    """Chat消息选择器"""
    index = fields.Int(required=True)
    message = fields.Nested(nested=ChatMessageSchema)  # noqa
    finish_reason = fields.Str(validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]))  # noqa


class ChatChoiceChunkSchema(Schema):
    """Chat流式消息选择器"""
    index = fields.Int(required=True)
    delta = fields.Nested(nested=ChatMessageSchema)  # noqa
    finish_reason = fields.Str(validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]))  # noqa


class ChatResponseSchema(Schema):
    """Chat接口响应数据结构映射"""
    id = fields.Str(dump_default=lambda: uuid4().hex)
    created = fields.Int(dump_default=lambda: int(time()))
    model = fields.Str(required=True)
    choices = fields.List(fields.Nested(nested=ChatChoiceSchema), required=True)  # noqa
    object = fields.Constant(constant="chat")


class ChatResponseChunkSchema(Schema):
    """Chat接口流式响应数据结构映射"""
    id = fields.Str(dump_default=lambda: uuid4().hex)
    created = fields.Int(dump_default=lambda: int(time()))
    model = fields.Str(required=True)
    choices = fields.List(fields.Nested(nested=ChatChoiceChunkSchema), required=True)  # noqa
    object = fields.Constant(constant="chat")


# # Completions接口
#
# class CompletionsRequestSchema(Schema):
#     model = fields.Str(required=True)
#     prompt = fields.Raw(metadata={"example": "Say this is a test"})
#     stream = fields.Bool(load_default=False)
#     max_tokens = fields.Int(load_default=1024)
#     n = fields.Int(load_default=1)
#     seed = fields.Int(load_default=1)
#     top_p = fields.Float(load_default=1.0)
#     temperature = fields.Float(load_default=1.0)
#     presence_penalty = fields.Float(load_default=0.0)
#     frequency_penalty = fields.Float(load_default=0.0)
#     logit_bias = fields.Dict(load_default=None)  # noqa
#
#
# class CompletionChoiceSchema(Schema):
#     index = fields.Int(required=True)
#     text = fields.Str(required=True)
#     logprobs = fields.Dict(load_default=None)  # noqa
#     finish_reason = fields.Str(validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]))  # noqa
#
#
# class CompletionsResponseSchema(Schema):
#     id = fields.Str(dump_default=lambda: uuid4().hex)
#     created = fields.Int(dump_default=lambda: int(time()))
#     model = fields.Str(required=True)
#     choices = fields.List(fields.Nested(nested=CompletionChoiceSchema), required=True)  # noqa
#     object = fields.Constant(constant="completions")


# Embeddings接口

class EmbeddingsRequestSchema(Schema):
    """Embeddings接口请求数据结构解析"""
    model = fields.Str(required=True)  # noqa
    input = fields.List(fields.Str, required=True)  # noqa


class EmbeddingsDataSchema(Schema):
    """Embeddings结果数据"""
    index = fields.Int(required=True)
    embedding = fields.List(fields.Float, required=True)  # noqa
    object = fields.Constant(constant="embedding")


class EmbeddingsUsageSchema(Schema):
    """Embeddings用量数据"""
    prompt_tokens = fields.Int(required=True)
    total_tokens = fields.Int(required=True)


class EmbeddingsResponseSchema(Schema):
    """Embeddings接口响应数据结构映射"""
    model = fields.Str(required=True)  # noqa
    data = fields.List(fields.Nested(nested=EmbeddingsDataSchema))  # noqa
    usage = fields.Nested(nested=EmbeddingsUsageSchema)  # noqa
    object = fields.Constant(constant="embeddings")
