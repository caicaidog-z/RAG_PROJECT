from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
from uuid import uuid4


JobStatus = Literal["queued", "running", "completed", "failed"]
JobMode = Literal["path", "upload"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    id: str
    mode: JobMode
    source: str
    files: list[str]
    file_count: int
    status: JobStatus = "queued"
    message: str = "等待执行"
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self, max_jobs: int = 50):
        self.max_jobs = max_jobs
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._lock = Lock()

    def create_job(self, mode: JobMode, source: str, files: list[str]) -> JobRecord:
        with self._lock:
            job = JobRecord(
                id=uuid4().hex[:12],
                mode=mode,
                source=source,
                files=files,
                file_count=len(files),
            )
            self._jobs[job.id] = job
            self._trim_if_needed()
            return job

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if message is not None:
                job.message = message
            job.error = error
            job.updated_at = utc_now_iso()
            self._jobs.move_to_end(job_id)
            return job

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def has_active_job(self) -> bool:
        with self._lock:
            return any(job.status in {"queued", "running"} for job in self._jobs.values())

    def _trim_if_needed(self) -> None:
        while len(self._jobs) > self.max_jobs:
            self._jobs.popitem(last=False)
