from __future__ import annotations

import os

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from study_assistant import config


def build_default_chat_model():
    if not os.getenv("DASHSCOPE_API_KEY"):
        return None
    return ChatTongyi(model=config.CHAT_MODEL_NAME, temperature=0.2)


def build_default_embeddings():
    if not os.getenv("DASHSCOPE_API_KEY"):
        return DeterministicFakeEmbedding(size=64)
    return DashScopeEmbeddings(model=config.EMBEDDING_MODEL_NAME)
