# RAG Backend

一个基于 LangGraph、Milvus 和 LangChain 的 RAG 实验项目。当前主线流程已经整理到 `workflows/`，支持交互式问答和 Markdown 文档入库。

## 目录

- `config/`: 环境变量和运行配置
- `models/`: LLM 与 Embedding 初始化
- `ingestion/`: Markdown 解析与 Milvus 入库
- `retrieval/`: Milvus 连接与 retriever/tool
- `workflows/`: 主 RAG 工作流和节点
- `services/`: 对外统一服务接口
- `interfaces/`: CLI 入口
- `archive/experimental/`: 旧实验代码归档

## 环境变量

复制 `.env.example` 并填写：

```bash
cp .env.example .env
```

## 运行

交互式问答：

```bash
python main.py chat
```

Markdown 入库：

```bash
python main.py ingest <markdown_dir>
```

## 当前主线

- 主图编排：`workflows/rag_graph.py`
- 问答入口：`main.py` -> `interfaces/cli.py`
- 入库入口：`ingestion/milvus_ingest.py`
