from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path
from queue import Queue

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect, WebSocketState

from config.settings import ConfigurationError, get_settings
from ingestion.milvus_ingest import SUPPORTED_DOCUMENT_EXTENSIONS, ingest_directory
from services.qa_service import answer_question, stream_question
from webapp.job_store import JobRecord, JobStore
from webapp.schemas import (
    ChatAnswerResponse,
    ChatRequest,
    HealthResponse,
    JobListResponse,
    JobResponse,
    PathIngestRequest,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CHUNK_SIZE = 1024 * 1024

app = FastAPI(title="RAG Backend Console", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

job_store = JobStore()


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    milvus_status = "ok"
    milvus_detail = "connected"
    try:
        client = MilvusClient(uri=settings.milvus_uri)
        collections = client.list_collections()
        milvus_detail = f"{len(collections)} collections visible"
    except Exception as exc:  # noqa: BLE001
        milvus_status = "error"
        milvus_detail = str(exc)

    status = "ok" if milvus_status == "ok" else "degraded"
    return HealthResponse(
        status=status,
        collection_name=settings.collection_name,
        milvus_status=milvus_status,
        milvus_detail=milvus_detail,
        ocr_enabled=settings.pdf_enable_image_ocr,
        upload_dir=settings.web_upload_dir,
        active_job=job_store.has_active_job(),
    )


@app.get("/api/jobs", response_model=JobListResponse)
async def list_jobs() -> JobListResponse:
    return JobListResponse(jobs=[_to_job_response(job) for job in job_store.list_jobs()])


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _to_job_response(job)


@app.post("/api/chat", response_model=ChatAnswerResponse)
async def chat(request: ChatRequest) -> ChatAnswerResponse:
    try:
        answer = await run_in_threadpool(answer_question, request.question.strip())
    except (ConfigurationError, MilvusException) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatAnswerResponse(answer=answer)


@app.websocket("/ws/chat")
async def chat_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        question = str(payload.get("question", "")).strip()
        if not question:
            await websocket.send_json({"type": "error", "message": "问题不能为空"})
            return

        queue: Queue = Queue()
        worker = threading.Thread(
            target=_run_chat_stream,
            args=(question, queue),
            daemon=True,
        )
        worker.start()

        while True:
            event = await asyncio.to_thread(queue.get)
            if event is None:
                break
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass


@app.post("/api/ingest/path", response_model=JobResponse)
async def ingest_from_path(request: PathIngestRequest) -> JobResponse:
    settings = get_settings()
    source_path = Path(request.path).expanduser().resolve()
    if not request.confirm_reset:
        raise HTTPException(status_code=400, detail="请先确认会重建当前 collection")
    if not source_path.is_dir():
        raise HTTPException(status_code=400, detail="目录不存在")
    files = _list_supported_files(source_path)
    if not files:
        raise HTTPException(status_code=400, detail="目录顶层未发现 .md 或 .pdf 文件")
    _ensure_no_active_job()

    job = job_store.create_job("path", str(source_path), files)
    _start_ingest_job(job, source_path, settings.collection_name)
    return _to_job_response(job)


@app.post("/api/ingest/upload", response_model=JobResponse)
async def ingest_from_upload(
    files: list[UploadFile] = File(...),
    confirm_reset: bool = Form(False),
) -> JobResponse:
    settings = get_settings()
    if not confirm_reset:
        raise HTTPException(status_code=400, detail="请先确认会重建当前 collection")
    _ensure_no_active_job()
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文件")

    upload_dir = _build_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_files = await _save_upload_files(upload_dir, files)
    if not saved_files:
        raise HTTPException(status_code=400, detail="上传文件类型不支持，仅支持 .md 和 .pdf")

    job = job_store.create_job(
        "upload",
        str(upload_dir),
        [path.name for path in saved_files],
    )
    _start_ingest_job(job, upload_dir, settings.collection_name)
    return _to_job_response(job)


def serve() -> None:
    settings = get_settings()
    uvicorn.run(
        "webapp.app:app",
        host=settings.web_host,
        port=settings.web_port,
        reload=False,
    )


def _to_job_response(job: JobRecord) -> JobResponse:
    return JobResponse.model_validate(job.to_dict())


def _ensure_no_active_job() -> None:
    if job_store.has_active_job():
        raise HTTPException(status_code=409, detail="当前已有入库任务在执行")


def _build_upload_dir() -> Path:
    settings = get_settings()
    base_dir = Path(settings.web_upload_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"job_{threading.get_native_id()}_{int(asyncio.get_running_loop().time() * 1000)}"


def _list_supported_files(document_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in document_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    )


async def _save_upload_files(upload_dir: Path, uploads: list[UploadFile]) -> list[Path]:
    saved_files: list[Path] = []
    for index, upload in enumerate(uploads, start=1):
        original_name = Path(upload.filename or f"upload_{index}").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
            await upload.close()
            continue

        safe_name = _safe_filename(original_name, index)
        destination = upload_dir / safe_name
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
        await upload.close()
        saved_files.append(destination)
    return saved_files


def _safe_filename(filename: str, index: int) -> str:
    suffix = Path(filename).suffix
    stem = Path(filename).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or f"document_{index}"
    return f"{index:03d}_{safe_stem}{suffix}"


def _start_ingest_job(job: JobRecord, document_dir: Path, collection_name: str) -> None:
    worker = threading.Thread(
        target=_run_ingest_job,
        args=(job.id, document_dir, collection_name),
        daemon=True,
    )
    worker.start()


def _run_ingest_job(job_id: str, document_dir: Path, collection_name: str) -> None:
    job_store.update_job(
        job_id,
        status="running",
        message=f"正在写入 collection {collection_name}",
    )
    try:
        ingest_directory(str(document_dir))
    except Exception as exc:  # noqa: BLE001
        job_store.update_job(
            job_id,
            status="failed",
            message="入库失败",
            error=str(exc),
        )
        return

    job_store.update_job(
        job_id,
        status="completed",
        message=f"已完成，写入 collection {collection_name}",
        error=None,
    )


def _run_chat_stream(question: str, queue: Queue) -> None:
    final_answer = ""
    try:
        for output in stream_question(question):
            for node_name, state in output.items():
                event = {
                    "type": "node",
                    "node": node_name,
                    "question": state.get("question", question),
                }
                documents = _normalize_documents(state.get("documents"))
                if documents:
                    event["documents"] = [_serialize_document(doc) for doc in documents]
                if "generation" in state:
                    event["generation"] = state.get("generation", "")
                    final_answer = state.get("generation", final_answer)
                queue.put(event)
        queue.put({"type": "final", "answer": final_answer})
    except (ConfigurationError, MilvusException) as exc:
        queue.put({"type": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        queue.put({"type": "error", "message": str(exc)})
    finally:
        queue.put(None)


def _normalize_documents(documents: object) -> list[Document]:
    if documents is None:
        return []
    if isinstance(documents, Document):
        return [documents]
    if isinstance(documents, list):
        return [doc for doc in documents if isinstance(doc, Document)]
    return []


def _serialize_document(document: Document) -> dict:
    return {
        "content": document.page_content,
        "metadata": dict(document.metadata),
    }
