# 第一阶段改动总结

## 目标

第一阶段的目标不是重写业务逻辑，而是先把仓库从“脚本堆”整理成“有主线、有入口、有目录分层”的结构，方便后续继续做配置治理、依赖收口和副作用消除。

本阶段完成的核心工作：

- 收拢主线流程
- 建立新的目录分层
- 增加统一 CLI 入口
- 归档旧实验代码
- 补充基础文档和配置样例
- 扩充基础忽略规则

## 目录结构改造

### 新增目录

- `config/`
- `models/`
- `ingestion/`
- `retrieval/`
- `workflows/`
- `workflows/nodes/`
- `services/`
- `interfaces/`
- `scripts/`
- `docs/`
- `tests/`
- `archive/experimental/`

### 主线迁移

原 `graph2/` 主流程迁移为新的工作流主线：

- `graph2/graph_2.py` -> `workflows/rag_graph.py`
- `graph2/graph_state2.py` -> `workflows/state.py`
- `graph2/retriever_node.py` -> `workflows/nodes/retrieve.py`
- `graph2/grade_documents_node.py` -> `workflows/nodes/grade_documents.py`
- `graph2/generate_node2.py` -> `workflows/nodes/generate_answer.py`
- `graph2/transform_query_node.py` -> `workflows/nodes/rewrite_query.py`
- `graph2/web_search_node.py` -> `workflows/nodes/web_search.py`
- `graph2/query_route_chain.py` -> `workflows/nodes/route_question.py`
- `graph2/grade_answer_chain.py` -> `workflows/nodes/grade_answer.py`
- `graph2/grade_hallucinations_chain.py` -> `workflows/nodes/grade_hallucination.py`
- `graph2/grader_chain.py` -> `workflows/nodes/document_grader_chain.py`

### 基础模块迁移

- `utils/env_utils.py` -> `config/settings.py`
- `llm_models/all_llm.py` -> `models/llm_factory.py`
- `llm_models/embeddings_model.py` -> `models/embedding_factory.py`
- `documents/markdown_parser.py` -> `ingestion/markdown_parser.py`
- `documents/write_milvus.py` -> `ingestion/milvus_ingest.py`
- `documents/milvus_db.py` -> `retrieval/milvus_store.py`
- `tools/retriever_tools.py` -> `retrieval/retriever_tool.py`
- `utils/log_utils.py` -> `utils/logging.py`
- `utils/print_utils.py` -> `utils/graph_debug.py`
- `draw_png.py` -> `scripts/draw_graph.py`

### 实验代码归档

旧的实验实现已归档，不再作为主线使用：

- `graph/` -> `archive/experimental/graph/`
- `agent/rag_agent.py` -> `archive/experimental/agent/rag_agent.py`

## 代码层面的具体改动

### 1. 统一入口

新增统一 CLI：

- `main.py`
- `interfaces/cli.py`

当前支持的命令：

- `python main.py chat`
- `python main.py ingest <markdown_dir>`

CLI 做了最小错误处理：

- 没有参数时输出用法
- `ingest` 缺目录参数时输出用法
- `ingest` 目录不存在时返回明确错误

### 2. 主图编排与服务层收拢

- `workflows/rag_graph.py` 现在只负责定义和构建主 RAG 图
- 原先放在 `graph_2.py` 底部的交互式 `while True` 已移出工作流文件
- 新增 `services/qa_service.py`，对主图提供统一调用入口

### 3. 配置抽取

`config/settings.py` 已统一承接以下配置：

- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `OPENAI_BASE_URL`
- `MODEL_NAME`
- `MILVUS_URI`
- `COLLECTION_NAME`

相比改造前，`MILVUS_URI`、`COLLECTION_NAME`、`MODEL_NAME`、`OPENAI_BASE_URL` 已支持从环境变量读取，而不是全部写死。

### 4. 入库流程整理

`ingestion/milvus_ingest.py` 已从硬编码脚本调整为可复用函数：

- 新增 `ingest_directory(md_dir, queue_maxsize=20)`
- 去掉了原入口中的固定 Windows 路径依赖
- 目录检查前置到了运行逻辑里

### 5. 文档与配置样例

新增：

- `README.md`
- `docs/architecture.md`
- `.env.example`

### 6. 忽略规则补齐

`.gitignore` 已补充：

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `.idea/`
- `.env`
- `logs/`

## 当前已经实现的功能

以下能力在第一阶段已经落地：

### 结构与入口能力

- 已有明确主线目录结构
- 已有统一入口 `main.py`
- 已有 CLI 命令分发
- 已有实验代码归档区
- 已有主线工作流路径和架构文档

### 文档入库能力

- 可以通过 `python main.py ingest <markdown_dir>` 触发 Markdown 入库流程
- 入库逻辑仍保留原有的多进程解析 + 写入 Milvus 机制
- Markdown 解析、标题合并、语义切块能力仍在

### 问答编排能力

- 主 RAG 图已经收口到 `workflows/rag_graph.py`
- 仍保留“问题路由 -> 检索/网页搜索 -> 文档评分 -> 生成 -> 回答/幻觉评分 -> 重试”的主流程
- CLI 侧可以通过 `python main.py chat` 进入问答模式

### 配置与文档能力

- 已支持 `.env` 风格的基础配置读取
- 已提供 `.env.example`
- 已提供 README 和架构说明

### 基础验证

本阶段已完成过的最小验证：

- 关键文件 `py_compile` 通过
- `python main.py` 可正常输出用法
- `python main.py ingest <不存在目录>` 可返回明确错误

## 当前没有实现，或只实现了一半的功能

下面这些能力还没有真正做完，或者只做了结构层整理，没有做工程化闭环。

### 1. 导入副作用没有彻底消除

虽然 CLI 已做延迟导入，避免了“只看帮助信息就初始化外部依赖”的问题，但项目内部仍存在模块级初始化：

- `models/llm_factory.py` 中 `llm` 是全局实例
- `models/embedding_factory.py` 中 embedding 是全局实例
- `retrieval/retriever_tool.py` 中 retriever / retriever_tool 是全局实例
- `services/qa_service.py` 中直接引用了模块级 `graph`

这意味着：

- 真正进入 `chat` 或 `ingest` 时仍会触发外部依赖初始化
- 还不适合做可控测试
- 还不适合直接扩展成稳定服务

### 2. 配置治理没有完成

当前只是把配置聚合到了 `config/settings.py`，但还没有：

- 用 `pydantic-settings` 做配置对象化
- 做必填项校验
- 做配置默认值与运行环境分层
- 区分开发、测试、生产环境

### 3. 检索层职责仍然耦合

`retrieval/retriever_tool.py` 仍然同时负责：

- 建 Milvus 连接
- 构造 retriever
- 包装 LangChain tool

这一层还没有拆成工厂函数和显式依赖注入。

### 4. 主图的循环控制没有加强

`workflows/rag_graph.py` 仍保留原先的流程风险：

- `not supported -> generate` 可能重复重试
- 没有最大生成重试次数
- 没有统一失败出口
- 没有稳定的异常恢复策略

### 5. Prompt 与评分链还未抽象

当前各节点内部仍然散落着 prompt / schema / grading 逻辑，还没有抽成：

- `workflows/prompts.py`
- `workflows/schemas.py`

这会影响后续 prompt 调优和测试。

### 6. 服务化没有实现

当前只有 CLI，没有：

- FastAPI / HTTP API
- 会话管理
- 监控与 tracing
- 错误码与接口契约
- 服务级启动方式

### 7. 测试没有实现

虽然已经建了 `tests/` 目录，但目前没有真正落地的测试用例：

- 没有 parser 单测
- 没有 graph 路由测试
- 没有 mock LLM / mock Milvus 的集成测试

### 8. 仍有遗留命名与实现问题

例如：

- `MilvusVectorSave` 命名仍偏旧
- `utils/logging.py` 名称容易和标准库 `logging` 混淆
- `retrieval/milvus_store.py` 仍保留 `__main__` 调试代码
- `ingestion/milvus_ingest.py` 的异常处理和状态返回还比较粗糙

## 当前功能边界结论

如果按“现在这个仓库能否被别人拿来继续开发”来评估：

- **已完成**：主线结构整理、入口统一、基础文档、实验代码归档、命令入口和最小运行验证
- **部分完成**：问答与入库流程的工程化封装
- **未完成**：副作用治理、配置对象化、测试体系、服务化、流程稳定性增强

换句话说，第一阶段已经把项目从“难以看懂”推进到了“能顺着结构继续改”，但还没有推进到“稳定可维护、可上线”的程度。

## 建议作为第二阶段的起点

建议第二阶段优先处理这三件事：

1. 把 LLM / embedding / retriever / graph 改成工厂函数，彻底消除 import 副作用
2. 把 `config/settings.py` 升级为配置对象，并做必填校验
3. 给 `workflows/rag_graph.py` 增加明确的重试上限和失败出口
