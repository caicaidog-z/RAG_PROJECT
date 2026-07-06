"""依赖注入：惰性单例，避免启动期连接失败。

设计说明：
- `tools/retriever_tools` 在 import 时就会连接 Milvus（模块级副作用），
  这里用函数级懒加载把连接推迟到首次问答请求，使 `uvicorn api.app:app`
  能在 Milvus 暂未就绪时也完成启动。
- 同步问答接口需要 `graph2.graph_2.graph`（编译后的 LangGraph），
  同样惰性获取；import 该模块本身不再触发 REPL（已重构进 __main__）。
"""

from functools import lru_cache

from documents.milvus_db import MilvusVectorSave
from utils.env_utils import COLLECTION_NAME
from utils.log_utils import log


@lru_cache(maxsize=1)
def get_milvus_store() -> MilvusVectorSave:
    """返回已连接的 MilvusVectorSave 单例（入库与按文件名删除共用）。"""
    store = MilvusVectorSave()
    store.create_connection()
    log.info("MilvusVectorSave 连接已建立")
    return store


@lru_cache(maxsize=1)
def get_retriever():
    """返回 Milvus hybrid 检索器单例。

    直接 import `tools.retriever_tools.retriever` 会触发模块级 Milvus 连接，
    这里包一层以保持调用点语义清晰；import 仍只发生一次（Python 模块缓存）。
    """
    from tools.retriever_tools import retriever  # noqa: WPS433 - 惰性 import
    log.info("Milvus 检索器已加载")
    return retriever


@lru_cache(maxsize=1)
def get_graph():
    """返回编译后的 LangGraph 单例（供同步问答接口使用）。"""
    from graph2.graph_2 import graph  # noqa: WPS433 - 惰性 import
    log.info("LangGraph 工作流已编译并加载")
    return graph


@lru_cache(maxsize=1)
def get_llm():
    """返回 ChatOpenAI LLM 单例（流式问答直接 astream 调用）。"""
    from llm_models.all_llm import llm  # noqa: WPS433 - 惰性 import
    return llm


def get_collection_name() -> str:
    """当前 Milvus collection 名（按文件名删除时使用）。"""
    return COLLECTION_NAME
