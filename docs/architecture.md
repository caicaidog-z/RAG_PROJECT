# Architecture

当前项目的主链路如下：

1. `ingestion/markdown_parser.py` 解析 Markdown 并切块
2. `retrieval/milvus_store.py` 管理 Milvus collection 和连接
3. `retrieval/retriever_tool.py` 暴露 LangChain retriever/tool
4. `workflows/rag_graph.py` 编排路由、检索、评分、生成
5. `interfaces/cli.py` 提供 `chat` 和 `ingest` 命令

旧的 `graph/` 与 `agent/` 实验代码已归档到 `archive/experimental/`。
