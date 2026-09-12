"""
Simple in-memory job registry for long-running bulk operations (currently
just the Airtable bulk ingest) that need to report live progress to a
polling client without blocking the triggering HTTP request on the whole
operation.

In-memory, not persisted — matches this project's existing philosophy for
process-local state (see app/faiss_index.py). A job's progress is only
meaningful to whoever is currently watching it in the browser; if the
server restarts mid-job, that's an acceptable edge case for a manually
triggered bulk operation, not something worth persisting for.

Jobs are deliberately decoupled from the HTTP request that started them:
the actual work runs as a background asyncio task via fire_and_forget()
below, independent of whether a client is polling for progress. Closing
the browser tab mid-ingest does not stop or affect the ingestion itself.
"""
import asyncio
import time
import uuid
from typing import Coroutine, Dict, Optional

_jobs: Dict[str, dict] = {}

# asyncio.create_task() doesn't keep the task alive on its own — if nothing
# holds a reference, it can be garbage-collected mid-run. This set holds a
# reference until the task finishes, then drops it via the done-callback.
_background_tasks: set = set()


def fire_and_forget(coro: Coroutine) -> asyncio.Task:
    """Run a coroutine as a true background task, safe from GC, independent of any HTTP request's lifecycle."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def create_job(kind: str) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "kind": kind,
        "status": "running",  # "running" | "done" | "error"
        "current": None,
        "records_scanned": 0,
        "images_found": 0,
        "images_done": 0,
        "results": [],
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }
    return job_id


def update_job(job_id: str, **fields) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(fields)


def append_result(job_id: str, result: dict) -> None:
    if job_id in _jobs:
        _jobs[job_id]["results"].append(result)
        _jobs[job_id]["images_done"] = len(_jobs[job_id]["results"])


def finish_job(job_id: str, status: str = "done", error: Optional[str] = None) -> None:
    if job_id in _jobs:
        _jobs[job_id]["status"] = status
        _jobs[job_id]["error"] = error
        _jobs[job_id]["finished_at"] = time.time()


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


def list_jobs() -> list:
    """All known jobs (this process only, in-memory), most recently started first."""
    return sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
