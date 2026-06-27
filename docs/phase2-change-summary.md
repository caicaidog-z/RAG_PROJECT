# 第二阶段改动总结

## 目标

第二阶段主要解决两个问题：

- import 阶段就初始化 LLM / embedding / retriever / graph 的副作用
- 配置只是常量聚合，没有校验和运行期错误边界

## 本阶段完成的改动

### 1. 配置对象化

`config/settings.py` 已改为 `pydantic-settings` 驱动，新增：

- `AppSettings`
- `get_settings()`
- `ConfigurationError`

并将运行配置统一收口为可缓存、可校验的设置对象。

### 2. 模型与 embedding 懒加载

`models/llm_factory.py` 已改为：

- `build_llm()`
- `get_llm()`
- `build_web_search_tool()`
- `get_web_search_tool()`

`models/embedding_factory.py` 已改为：

- `build_openai_embedding()`
- `get_openai_embedding()`
- `build_bge_embedding()`
- `get_bge_embedding()`
- `get_chunking_embedding()`

这些对象现在都不会在 import 时初始化。

### 3. 检索层拆分

新增：

- `retrieval/retriever_factory.py`

现在职责分为：

- `retrieval/milvus_store.py`: Milvus collection / connection / add_documents
- `retrieval/retriever_factory.py`: 构建 retriever
- `retrieval/retriever_tool.py`: 构建 LangChain tool

### 4. 工作流链路按需构建

以下评分链和路由链都已改成 getter：

- `get_retrieval_grader_chain()`
- `get_answer_grader_chain()`
- `get_hallucination_grader_chain()`
- `get_question_router_chain()`

`workflows/rag_graph.py` 也不再在模块加载时直接创建 graph，而是：

- `build_graph()`
- `get_graph()`

### 5. 服务层去掉模块级 graph

`services/qa_service.py` 改为：

- `QAService`
- `get_qa_service()`
- `stream_question()`
- `answer_question()`

### 6. CLI 运行期错误更清晰

`interfaces/cli.py` 现在会对以下情况给出明确提示：

- 缺少目录参数
- 目录不存在
- 缺少运行配置
- Milvus 连接失败

## 当前验证结果

本阶段完成后：

- `python main.py` 可正常输出用法
- `python main.py chat` 不会在 import 阶段报错
- 当缺少 `OPENAI_API_KEY` 时，`chat` 会在真正调用模型时返回明确配置错误
- `python main.py ingest <dir>` 不再因为缺少 `OPENAI_API_KEY` 在 embedding 初始化阶段失败
- `ingest` 现在会继续走到 Milvus 连接阶段；如果 Milvus 不可用，会返回明确的 Milvus 错误

## 仍未完成的事项

- 主图的生成重试上限和失败出口还未增强
- Prompt / schema 还没有抽成集中模块
- 还没有测试用例
- 还没有 API 服务化
