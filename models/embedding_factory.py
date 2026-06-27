from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from config.settings import AppSettings, get_settings


def build_openai_embedding(settings: AppSettings | None = None) -> OpenAIEmbeddings:
    settings = settings or get_settings()
    api_key = settings.require_openai_embedding_key()
    return OpenAIEmbeddings(
        openai_api_key=api_key,
        openai_api_base=settings.openai_base_url,
    )


@lru_cache(maxsize=1)
def get_openai_embedding() -> OpenAIEmbeddings:
    return build_openai_embedding()


def build_bge_embedding(settings: AppSettings | None = None) -> HuggingFaceEmbeddings:
    settings = settings or get_settings()
    model_kwargs = {"device": settings.bge_device}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=settings.bge_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


@lru_cache(maxsize=1)
def get_bge_embedding() -> HuggingFaceEmbeddings:
    return build_bge_embedding()


def get_chunking_embedding(settings: AppSettings | None = None):
    settings = settings or get_settings()
    if settings.chunking_embedding_provider == "openai":
        return get_openai_embedding()
    return get_bge_embedding()
