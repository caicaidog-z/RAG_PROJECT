from functools import lru_cache

from config.settings import AppSettings, get_settings
from retrieval.milvus_store import MilvusStore


def build_retriever(settings: AppSettings | None = None):
    settings = settings or get_settings()
    store = MilvusStore(settings=settings)
    vector_store = store.ensure_connection()
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.retriever_k,
            "score_threshold": settings.retriever_score_threshold,
            "ranker_type": "rrf",
            "ranker_params": {"k": settings.retriever_ranker_k},
            "filter": {"category": settings.retriever_category_filter},
        },
    )


@lru_cache(maxsize=1)
def get_retriever():
    return build_retriever()
