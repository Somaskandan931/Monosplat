"""
backend/app/workers/job_runner.py
----------------------------------
Lightweight async job runner for MonoSplat.
"""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from backend.app.models.orm import Job

log = logging.getLogger("monosplat.workers")

_executor = ProcessPoolExecutor(max_workers=2)


def create_job(
    db: Session,
    *,
    job_type: str,
    project_id: Optional[str] = None,
    training_run_id: Optional[str] = None,
) -> Job:
    """Persist a new Job row with status=pending and return it."""
    job = Job(
        job_type=job_type,
        project_id=project_id,
        training_run_id=training_run_id,
        status="pending",
        progress=0.0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_job(
    db: Session,
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
) -> Optional[Job]:
    """Partial-update a Job row."""
    job = db.get(Job, job_id)
    if job is None:
        return None
    if status      is not None: job.status      = status
    if progress    is not None: job.progress     = progress
    if message     is not None: job.message      = message
    if result      is not None: job.result       = result
    if error       is not None: job.error        = error
    if started_at  is not None: job.started_at   = started_at
    if finished_at is not None: job.finished_at  = finished_at
    db.commit()
    db.refresh(job)
    return job


def _run_task_in_subprocess(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return fn(**kwargs)
    except Exception as exc:
        return {"__error__": str(exc), "__tb__": traceback.format_exc()}


def submit_background_job(
    db: Session,
    *,
    job_id: str,
    fn: Callable,
    kwargs: Dict[str, Any],
    db_factory: Callable[[], Session],
) -> None:
    future = _executor.submit(_run_task_in_subprocess, fn, kwargs)

    def _on_done(fut):
        new_db = db_factory()
        try:
            result = fut.result()
            if isinstance(result, dict) and "__error__" in result:
                update_job(
                    new_db, job_id,
                    status="failed",
                    progress=100.0,
                    error=result["__error__"],
                    finished_at=datetime.utcnow(),
                )
                log.error("Job %s failed:\n%s", job_id, result.get("__tb__", ""))
            else:
                update_job(
                    new_db, job_id,
                    status="success",
                    progress=100.0,
                    result=result,
                    finished_at=datetime.utcnow(),
                )
                log.info("Job %s completed successfully", job_id)
        except Exception as exc:
            update_job(
                new_db, job_id,
                status="failed",
                error=str(exc),
                finished_at=datetime.utcnow(),
            )
        finally:
            new_db.close()

    update_job(db, job_id, status="running", started_at=datetime.utcnow())
    future.add_done_callback(_on_done)