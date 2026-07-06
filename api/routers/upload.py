"""文档上传路由：Markdown 文件 + Confluence 文档 ID。"""

import os
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import (
    ConfluenceRequest,
    ConfluenceUploadResponse,
    UploadMdResponse,
)
from api.services import ingest_service
from utils.log_utils import log

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/md", response_model=UploadMdResponse)
async def upload_md(file: UploadFile = File(...)):
    """上传 .md 文件到知识库。

    流程：保存到项目 md/ 目录 → 解析 → 按文件名查重删除旧向量 → 入库。
    """
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="仅支持 .md 文件")

    os.makedirs(ingest_service.MD_DIR, exist_ok=True)
    # 防路径穿越：只取文件名
    safe_name = os.path.basename(file.filename)
    save_path = os.path.join(ingest_service.MD_DIR, safe_name)
    with open(save_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    log.info(f"[upload_md] 已保存上传文件: {save_path}")

    filename, chunks = ingest_service.ingest_markdown_file(save_path)
    return UploadMdResponse(
        filename=filename,
        chunks=chunks,
        message=f"成功入库 {chunks} 个 chunk" if chunks else "解析后无 chunk 入库",
    )


@router.post("/confluence", response_model=ConfluenceUploadResponse)
async def upload_confluence(req: ConfluenceRequest):
    """传入 Confluence 文档 ID，抓取并入库。

    流程：ConfluenceFetcher 抓取 → 落盘 md → 解析 → 按文件名查重删除 → 入库。
    """
    try:
        filename, chunks = ingest_service.ingest_confluence(
            req.content_id, filename=req.filename
        )
    except ValueError as exc:
        # 未配置 CONFLUENCE_TOKEN
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception(f"[upload_confluence] 抓取/入库失败: {exc}")
        raise HTTPException(status_code=502, detail=f"Confluence 抓取/入库失败: {exc}")

    return ConfluenceUploadResponse(
        content_id=req.content_id,
        filename=filename,
        chunks=chunks,
        message=f"成功入库 {chunks} 个 chunk" if chunks else "解析后无 chunk 入库",
    )
