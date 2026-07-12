# RAG 企业知识库

基于 **Milvus 混合检索 + LangGraph 工作流** 的企业级 RAG（检索增强生成）知识库系统，面向「畅捷通新生产部门」业务领域。支持 Markdown / Confluence 文档入库、流式问答、网络搜索兜底，并提供完整的检索质量校验链路（文档相关性打分、幻觉检测、答案打分、问题重写重试）。

---

## ✨ 特性

- **混合检索**：Milvus dense（BGE 向量）+ sparse（内置 BM25），RRF 融合排序，中文用 jieba 分词
- **LangGraph 编排**：路由 → 检索 → 文档打分 → 生成 → 幻觉/答案校验 → 失败重写重试 / 网络搜索兜底
- **流式问答**：SSE 逐 token 推送，打字机效果；另有同步接口走完整质量校验
- **智能路由**：业务问题走向量库，时效/外部问题走网络搜索（Tavily）
- **文档分片**：固定大小 + 重合区间流式切分，表格完整保留不截断，按 UTF-8 字节控制 Milvus 上限
- **多源入库**：本地 Markdown 上传 + Confluence 文档抓取，按文件名查重覆盖
- **Web 界面**：内置单页前端，上传文档 + 聊天一体
- **惰性加载**：embedding/连接/graph 全部 `lru_cache` 惰性单例，uvicorn 秒级启动

---

## 🏗️ 架构概览

```
用户提问
  │
  ▼
┌─────────────────┐
│  路由 route_question  │  (LLM 结构化输出)
└────────┬────────┘
   vectorstore │        │ web_search
        ▼              ▼
   ┌──────────┐   ┌──────────┐
   │ retrieve │   │web_search│  (Tavily)
   │ (Milvus  │   └────┬─────┘
   │  hybrid) │        │
   └────┬─────┘        │
        ▼              │
   ┌──────────────┐    │
   │grade_documents│   │  (并发打分, 过滤无关文档)
   └──────┬───────┘    │
          ▼            │
   ┌────────────────┐  │
   │decide_to_generate│ │  无相关文档 → transform_query(≤2次) / web_search 兜底
   └──────┬─────────┘  │
          ▼            ▼
        ┌──────────────┐
        │   generate   │  (流式生成)
        └──────┬───────┘
               ▼
   ┌────────────────────────────┐
   │ grade(幻觉 + 答案相关性)      │
   │ useful→END / not supported→重试 / not useful→重写 │
   └────────────────────────────┘
```

> 完整架构图见根目录 `graph_rag2.png`。

---

## 📁 项目结构

```
RAG_PROJECT/
├── api/                  # FastAPI Web 服务
│   ├── app.py            # 应用工厂 + 静态文件挂载
│   ├── routers/          # 路由：chat（流式/同步）、upload（md/confluence）
│   ├── services/         # 业务：qa_service、ingest_service
│   ├── deps.py           # 惰性单例依赖注入
│   ├── schemas.py        # Pydantic 请求/响应模型
│   └── static/           # 单页前端 index.html
├── documents/            # 文档处理与向量库
│   ├── markdown_parser.py # Markdown 流式固定大小分片（表格保护）
│   ├── milvus_db.py      # Milvus 连接/建表/入库
│   ├── write_milvus.py   # 多进程批量入库
│   └── confluence_fetcher.py # Confluence REST 抓取
├── graph2/               # LangGraph RAG 工作流
│   ├── graph_2.py        # 图编译入口（含 CLI REPL）
│   ├── graph_state2.py   # GraphState 定义
│   ├── query_route_chain.py    # 路由
│   ├── retriever_node.py       # 检索节点
│   ├── grade_documents_node.py # 文档打分节点
│   ├── grader_chain.py         # 文档相关性打分链
│   ├── grade_hallucinations_chain.py # 幻觉检测
│   ├── grade_answer_chain.py   # 答案打分
│   ├── generate_node2.py       # 生成节点
│   ├── transform_query_node.py # 问题重写
│   └── web_search_node.py      # 网络搜索
├── llm_models/           # 模型工厂
│   ├── all_llm.py        # DeepSeek LLM + Tavily 搜索
│   └── embeddings_model.py # BGE embedding（惰性加载）
├── tools/
│   └── retriever_tools.py # Milvus hybrid retriever 封装
├── utils/                # env_utils / log_utils
├── md/                   # 上传/抓取的 Markdown 落盘目录
├── datas/                # 示例数据（被 .gitignore 忽略）
├── requirements.txt
├── .env.example          # 环境变量模板
└── .gitignore
```

---

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- **Milvus 2.5.x**（外部依赖，需自行启动；推荐 docker-compose）
- 可选：[Attu](https://github.com/zilliztech/attu)（Milvus 可视化管理）

### 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `torch` / `transformers` 体积较大，首次安装需一定时间。BGE embedding 模型（`BAAI/bge-small-zh-v1.5`，约 100MB）会在首次问答时自动从 HuggingFace 下载并缓存。

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

| 变量 | 必填 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | ✅ | [DeepSeek](https://platform.deepseek.com) API Key，当前 LLM 使用 |
| `MILVUS_URL` | ✅ | Milvus 地址，如 `http://localhost:19530` |
| `TAVILY_API_KEY` | ✅ | [Tavily](https://tavily.com) 搜索 Key（新用户免费额度） |
| `OPENAI_API_KEY` | ⬜ | 切换到 OpenAI 兼容网关时填写（当前默认走 DeepSeek） |
| `MODEL` / `BASE_URL` | ⬜ | 切 OpenAI 兼容网关时的模型名与地址 |
| `CONFLUENCE_BASE_URL` | ⬜ | Confluence 站点，默认 `https://wiki2.rd.chanjet.com` |
| `CONFLUENCE_TOKEN` | ⬜ | 仅抓取 Confluence 文档时需要 |

> ⚠️ `.env` 已在 `.gitignore` 中，**切勿提交真实密钥**。

### 4. 启动 Milvus（如未启动）

推荐用官方 docker-compose，启动后确认 `19530` 端口可用。可选启动 Attu：

```bash
docker run -d --name attu --network milvus \
  -p 8000:3000 -e MILVUS_URL=milvus-standalone:19530 zilliz/attu:v2.5
```

### 5. 启动服务

```bash
python -m uvicorn api.app:app --reload --port 8001
```

- 前端页面：http://localhost:8001/
- OpenAPI 文档：http://localhost:8001/docs
- 健康检查：http://localhost:8001/api/health

> 端口使用 `8001`（`8000` 留给 Attu）。首次问答会触发 BGE 模型加载（约 10-20 秒），之后正常。

---

## 📖 使用方式

### Web 界面

浏览器打开 http://localhost:8001/ ，左侧聊天、右侧上传文档。

- **发送**：流式问答（逐字输出，附来源）
- **同步问答**：走完整 LangGraph（含幻觉打分/重试，答案经质量校验，较慢）
- **上传 Markdown**：选择 `.md` 文件入库（按文件名查重覆盖）
- **Confluence 入库**：填文档 ID 抓取入库

### API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/chat/stream` | 流式问答（SSE），事件：`source` / `token` / `done` / `error` |
| `POST` | `/api/chat` | 同步问答（完整 LangGraph），返回 `{answer, sources[]}` |
| `POST` | `/api/upload/md` | 上传 Markdown 入库 |
| `POST` | `/api/upload/confluence` | 按 content_id 抓取 Confluence 文档入库 |
| `GET` | `/api/health` | 健康检查 |

流式问答示例：

```bash
curl -N -X POST http://localhost:8001/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "T+里怎么配置BOM和工艺路线?"}'
```

### 批量入库（CLI）

```bash
# 多进程批量写入 md/ 目录下所有 Markdown
python documents/write_milvus.py
```

---

## 🔧 关键技术细节

### 文档分片（`documents/markdown_parser.py`）

- **固定大小流式切分**：`chunk_size=500` 字符，`chunk_overlap=50` 字符重合区间
- **表格保护**：切点落在表格上时，把整个表格并入当前片，不从中间切断
- **字节上限控制**：所有长度判断用 UTF-8 字节数（Milvus `text` 字段 VARCHAR(6000)，中文每字符 3 字节），阈值 5500 字节留安全余量
- **超长混合片兜底**：含表格的片超字节上限时，拆成「文本段（按大小切）+ 表格段（按行拆 + 表头复用）」
- 元数据：`category`（content/Table）、`source`、`filename`、`filetype`、`title`、`category_depth`

### 检索（`tools/retriever_tools.py` + `documents/milvus_db.py`）

- **Milvus Hybrid**：dense（BGE 512 维，HNSW + IP）+ sparse（内置 BM25，jieba 分词）
- **RRF 融合**：`ranker_type="rrf"`，`k=4`，`score_threshold=0.1`
- **过滤**：`category in ['content', 'Table']`

### 两条问答链路（`api/services/qa_service.py`）

| | 流式 `/api/chat/stream` | 同步 `/api/chat` |
|---|---|---|
| 实现 | 自建异步链（`retriever.ainvoke` + `llm.astream`） | 复用编译后的 LangGraph |
| 质量校验 | 跳过（换取实时性） | 含幻觉打分/答案打分/重写重试 |
| 体验 | 逐 token 打字机 | 完整答案一次性返回，较慢 |

两者共用相同的 RAG Prompt，回答风格一致。

### 惰性加载（`api/deps.py`）

所有重资源（Milvus 连接、retriever、graph、LLM、embedding）均为 `@lru_cache` 惰性单例，首次问答才初始化，uvicorn 启动秒级完成。

---

## 🛠️ 技术栈

| 领域 | 选型 |
|---|---|
| 工作流编排 | LangGraph 0.3 |
| LLM | DeepSeek（`deepseek-chat`，OpenAI 兼容接口） |
| Embedding | BAAI/bge-small-zh-v1.5（HuggingFace，CPU） |
| 向量库 | Milvus 2.5（hybrid: dense + sparse/BM25） |
| Web 框架 | FastAPI + Uvicorn |
| 网络搜索 | Tavily |
| 文档处理 | unstructured / BeautifulSoup / markdown |
| 分词 | jieba（Milvus BM25 内置） |

---

## ⚠️ 注意事项

1. **密钥安全**：`.env` 不入库；如曾泄露 API Key，请立即到对应平台吊销重置
2. **模型下载**：BGE 模型首次加载需联网，已缓存后离线可用
3. **Milvus collection**：硬编码为 `t_collection01`，如需改名修改 `utils/env_utils.py`
4. **DeepSeek 配置**：`model` / `base_url` 在 `llm_models/all_llm.py` 硬编码，切 OpenAI 兼容网关需手动改
5. **CORS**：开发期全放行（`allow_origins=["*"]`），生产环境请收紧

---

## 📄 License

本项目仅供学习交流。
