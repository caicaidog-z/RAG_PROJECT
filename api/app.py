"""FastAPI 应用入口。

启动：
    python -m uvicorn api.app:app --reload --port 8001
访问：
    http://localhost:8001/         前端页面
    http://localhost:8001/docs     OpenAPI 调试

端口说明：使用 8001（8000 已被 Attu 占用）。
前置条件：Milvus 在 MILVUS_URL（默认 localhost:19530）运行；.env 已配置密钥。
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import chat, upload
from utils.log_utils import log

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def create_app() -> FastAPI:
    app = FastAPI(title="RAG 企业知识库 API", version="0.1.0")

    # CORS（开发期放行所有来源）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(upload.router)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # 静态前端放最后挂载：html=True 使访问 / 时返回 index.html。
    # 必须放在所有 /api/* 显式路由之后，否则 Mount("/") 会先吃掉 /api 请求。
    if os.path.isdir(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    else:
        log.warning(f"静态目录不存在: {STATIC_DIR}，前端不可用")

    return app


app = create_app()
