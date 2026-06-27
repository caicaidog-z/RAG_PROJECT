# RAG Backend

一个基于 LangGraph、Milvus 和 LangChain 的 RAG 实验项目。当前主线流程已经整理到 `workflows/`，支持交互式问答和 Markdown 文档入库。

第二阶段已完成懒加载重构：

- 不再在 import 阶段初始化 LLM、Embedding、Retriever、Graph
- 配置改为 `pydantic-settings` 驱动
- CLI 会对缺失配置和 Milvus 连接问题给出更明确的错误信息

## 目录

- `config/`: 环境变量和运行配置
- `models/`: LLM 与 Embedding 初始化
- `ingestion/`: Markdown/PDF 解析与 Milvus 入库
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

Web 控制台：

```bash
python main.py web
```

说明：

- 默认地址是 `http://127.0.0.1:8000`
- 页面内支持提问、查看检索过程、从目录入库、上传 `.md/.pdf` 入库
- `web` 仍然复用当前 `.env` 中的 `MILVUS_URI`、`COLLECTION_NAME`、OCR 配置等

交互式问答：

```bash
python main.py chat
```

说明：

- `chat` 需要有效的 `OPENAI_API_KEY`，或将 `MODEL_NAME` 切到 deepseek 并配置 `DEEPSEEK_API_KEY`
- `ingest` 默认使用本地 `bge` embedding 做切块，不再强依赖 `OPENAI_API_KEY`
- `ingest` 仍需要可连通的 `MILVUS_URI`

Markdown / PDF 入库：

```bash
python main.py ingest <document_dir>
```

说明：

- `ingest` 会扫描目录顶层的 `.md` 和 `.pdf`
- `.pdf` 会优先抽取原生文本；对嵌入图片会尝试使用本地 OCR，并把抽出的图片保存到 `logs/pdf_assets/`

## 当前主线

- 主图编排：`workflows/rag_graph.py`
- 问答入口：`main.py` -> `interfaces/cli.py`
- 入库入口：`ingestion/milvus_ingest.py`
- 检索构建：`retrieval/retriever_factory.py`
