"""文档入库编排：Markdown 解析 + 按文件名查重删除 + Milvus 写入。

复用：
- documents.markdown_parser.MarkdownParser
- documents.milvus_db.MilvusVectorSave（仅 create_connection + add_documents，
  绝不调 create_collection——它会 drop 整张表）
- documents.confluence_fetcher.ConfluenceFetcher
"""

import os

from documents.markdown_parser import MarkdownParser
from documents.milvus_db import MilvusVectorSave
from api.deps import get_collection_name, get_milvus_store
from utils.log_utils import log

# 与 ConfluenceFetcher 默认输出目录一致：项目根/md
MD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "md"
)


def _delete_by_filename(store: MilvusVectorSave, filename: str) -> int:
    """按 filename 先删后插的查重：删除该文件名的所有旧向量。

    通过 langchain_milvus.Milvus 内置的 pymilvus MilvusClient 执行
    `client.delete(collection_name, filter='filename == "xxx"')`。
    Milvus filter 表达式里字符串需用双引号包裹。
    """
    client = store.vector_store_saved.client
    collection = get_collection_name()
    # 先查数量用于日志
    try:
        existing = client.query(
            collection_name=collection,
            filter=f'filename == "{filename}"',
            output_fields=["id"],
        )
        old_count = len(existing)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[delete_by_filename] 查询旧记录失败（忽略）: {exc}")
        old_count = -1

    if old_count > 0:
        client.delete(
            collection_name=collection,
            filter=f'filename == "{filename}"',
        )
        log.info(f"[delete_by_filename] 已删除 {filename} 的旧向量 {old_count} 条")
    elif old_count == 0:
        log.info(f"[delete_by_filename] {filename} 无旧记录，直接新增")
    return max(old_count, 0)


def _ingest_markdown_path(md_file: str) -> tuple[str, int]:
    """解析单个 md 文件并入库，返回 (filename, chunk数)。

    流程：解析 → 按文件名查重删除 → add_documents。
    """
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(md_file)
    log.info(f"[ingest] 解析得到 {len(docs)} 个 chunk: {md_file}")

    if not docs:
        return os.path.basename(md_file), 0

    # 所有 chunk 的 filename metadata 应一致，取第一条
    filename = docs[0].metadata.get("filename", os.path.basename(md_file))

    store = get_milvus_store()
    _delete_by_filename(store, filename)
    store.add_documents(docs)
    log.info(f"[ingest] 入库完成: filename={filename}, chunks={len(docs)}")
    return filename, len(docs)


def ingest_markdown_file(md_file: str) -> tuple[str, int]:
    """对外：对已落盘的 md 文件执行入库。"""
    os.makedirs(MD_DIR, exist_ok=True)
    return _ingest_markdown_path(md_file)


def ingest_confluence(content_id: str, filename: str | None = None) -> tuple[str, int]:
    """对外：抓取 Confluence 文档 → 落盘 md → 入库。返回 (filename, chunk数)。"""
    # 延迟 import，避免未配置 CONFLUENCE_TOKEN 时启动即失败
    from documents.confluence_fetcher import ConfluenceFetcher

    fetcher = ConfluenceFetcher(output_dir=MD_DIR)
    md_path = fetcher.fetch_and_convert(content_id, filename=filename)
    log.info(f"[ingest_confluence] Confluence {content_id} -> {md_path}")
    return _ingest_markdown_path(md_path)
