from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_bge_embedding():
    """惰性加载 BGE embedding 模型（首次调用时才实例化，约 10-20 秒）。

    使用 lru_cache 保证全局单例，后续调用直接返回缓存对象。
    改为惰性后，import 本模块不再触发模型加载，uvicorn 启动秒级完成。
    """
    model_name = "BAAI/bge-small-zh-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )