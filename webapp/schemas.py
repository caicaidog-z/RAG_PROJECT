from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    collection_name: str
    milvus_status: Literal["ok", "error"]
    milvus_detail: str
    ocr_enabled: bool
    upload_dir: str
    active_job: bool


class PathIngestRequest(BaseModel):
    path: str
    confirm_reset: bool = False


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class JobResponse(BaseModel):
    id: str
    mode: Literal["path", "upload"]
    source: str
    files: list[str]
    file_count: int
    status: Literal["queued", "running", "completed", "failed"]
    message: str
    error: str | None = None
    created_at: str
    updated_at: str


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class ChatAnswerResponse(BaseModel):
    answer: str
