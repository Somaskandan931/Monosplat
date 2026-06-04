"""
backend/app/api/routes.py
--------------------------
MonoSplat REST API endpoints.

Flow
----
1. POST /upload                          — upload video, start preprocessing pipeline
2. GET  /status/{job_id}                 — poll job status + progress
3. GET  /download/{job_id}/colab-package — download Colab training ZIP
4. POST /upload-results/{job_id}         — import Colab training results ZIP
5. GET  /results/{job_id}                — get viewer file paths (ply / splat)
6. GET  /projects                        — list projects
"""

from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database.session import SessionLocal, get_db
from backend.app.models.orm import Job
from backend.app.workers.job_runner import create_job, submit_background_job
from backend.app.services.project_service import DatasetService

router = APIRouter()

_UPLOAD_ROOT  = Path("data/uploads")
_OUTPUTS_ROOT = Path("data/outputs")
_RESULTS_ROOT = Path("data/results")
_executor     = ProcessPoolExecutor(max_workers=2)

_ds_svc = DatasetService()


# ---------------------------------------------------------------------------
# Background task functions (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _pipeline_task(video_path: str, job_id: str, repo_root: str) -> dict:
    """Run full preprocessing pipeline in subprocess."""
    import sys
    from pathlib import Path as P

    for _p in (str(P(repo_root) / "src"), repo_root):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from scripts.pipeline import run_pipeline
    return run_pipeline(
        video_path=video_path,
        output_root=str(P(repo_root) / "data" / "outputs"),
    )


# ---------------------------------------------------------------------------
# 1. Upload video → start pipeline
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    file: UploadFile = File(...),
    project_name: str = Form("default"),
    db: Session = Depends(get_db),
):
    """Upload a video file and start the preprocessing pipeline."""
    upload_dir = _UPLOAD_ROOT
    upload_dir.mkdir(parents=True, exist_ok=True)

    job = create_job(db, job_type="pipeline")
    dest = upload_dir / f"{job.id}_{file.filename}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    repo_root = str(Path(__file__).resolve().parents[4])

    submit_background_job(
        db,
        job_id=job.id,
        fn=_pipeline_task,
        kwargs=dict(video_path=str(dest), job_id=job.id, repo_root=repo_root),
        db_factory=SessionLocal,
    )

    # Return all fields that UploadResponse / frontend types/api.ts expect
    return {
        "job_id":       job.id,
        "project_id":   project_name,          # logical project name used as ID until DB record created
        "dataset_id":   None,
        "upload_path":  str(dest),
        "frames_path":  None,                   # populated later by pipeline
        "filename":     file.filename,
        "size_bytes":   dest.stat().st_size,
        "message":      "Pipeline started. Poll /status/{job_id} for progress.",
    }


# ---------------------------------------------------------------------------
# 2. Poll job status
# ---------------------------------------------------------------------------

@router.get("/status/{job_id}")
def get_status(job_id: str, db: Session = Depends(get_db)):
    """Poll job status: pending → running → success / failed."""
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return {
        "job_id":       job.id,
        "job_type":     job.job_type,
        "status":       job.status,
        "progress":     job.progress,
        "message":      job.message,
        "result":       job.result,
        "error":        job.error,
        "created_at":   job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at),
        "started_at":   job.started_at.isoformat()  if job.started_at  else None,
        "finished_at":  job.finished_at.isoformat() if job.finished_at else None,
    }


# ---------------------------------------------------------------------------
# 3. Download Colab training package ZIP
# ---------------------------------------------------------------------------

@router.get("/download/{job_id}/colab-package")
def download_colab_package(job_id: str, db: Session = Depends(get_db)):
    """
    Stream the preprocessed Colab training ZIP for a completed pipeline job.
    The ZIP contains frames/ and sparse_text/ — drop it straight into Colab.
    """
    job = db.get(Job, job_id)
    candidate: Path | None = None

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    if job.status != "success":
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline job {job_id!r} has status '{job.status}' — ZIP is only available after success.",
        )

    if job.result and isinstance(job.result, dict):
        zip_from_result = job.result.get("zip")
        if zip_from_result:
            p = Path(zip_from_result)
            if p.exists():
                candidate = p

    # Fallback: scan outputs directory for any zip
    if candidate is None:
        for zip_file in sorted(_OUTPUTS_ROOT.rglob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
            candidate = zip_file
            break

    if candidate is None or not candidate.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Colab package ZIP not found for job {job_id!r}. "
                "The output may have been cleaned up."
            ),
        )

    return FileResponse(
        path=str(candidate),
        media_type="application/zip",
        filename="colab_training_package.zip",
    )


# ---------------------------------------------------------------------------
# 4. Import Colab training results  ← NEW
# ---------------------------------------------------------------------------

@router.post("/upload-results/{job_id}", status_code=status.HTTP_200_OK)
async def upload_results(
    job_id: str,
    file: UploadFile = File(...),
):
    """
    Accept a results ZIP exported from Colab after gsplat training.

    Expected ZIP layout (minimum):
        exports/final.ply
        exports/final.splat   (optional)

    The ZIP is extracted to data/results/{job_id}/ so that
    GET /results/{job_id} and the /static/results/ StaticFiles mount
    can immediately serve the viewer files.
    """
    from backend.app.services.result_service import import_results

    # Write upload to a temp file first — avoids holding the whole ZIP in RAM
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = import_results(zip_path=tmp_path, job_id=job_id)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not result.get("ply_path") and not result.get("splat_path"):
        raise HTTPException(
            status_code=422,
            detail=(
                "Results ZIP was extracted but no .ply or .splat file was found. "
                "Ensure your Colab export cell ran successfully (Cell 11)."
            ),
        )

    return {
        "import_job_id": job_id,
        "status":        "success",
        "message":       "Results imported. Open the Viewer tab and enter this job ID.",
        "ply_url":       f"/static/results/{job_id}/exports/final.ply"   if result.get("ply_path")   else None,
        "splat_url":     f"/static/results/{job_id}/exports/final.splat" if result.get("splat_path") else None,
    }


# ---------------------------------------------------------------------------
# 5. Get viewer paths
# ---------------------------------------------------------------------------

@router.get("/results/{job_id}")
def get_results(job_id: str):
    """Return viewer-ready file URLs for a completed job."""
    result_dir = _RESULTS_ROOT / job_id
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail="Results not found for this job")

    ply_files   = list(result_dir.rglob("*.ply"))
    splat_files = list(result_dir.rglob("*.splat"))

    return {
        "job_id":     job_id,
        "ply_url":    f"/static/results/{job_id}/exports/final.ply"   if ply_files   else None,
        "splat_url":  f"/static/results/{job_id}/exports/final.splat" if splat_files else None,
        "result_dir": str(result_dir),
    }


# ---------------------------------------------------------------------------
# 6. List projects (lightweight)
# ---------------------------------------------------------------------------

@router.get("/projects")
def list_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    from backend.app.services.experiment_service import ExperimentService
    projects = ExperimentService().list_projects(db, skip=skip, limit=limit)
    items = [
        {"id": p.id, "name": p.name,
         "created_at": p.created_at.isoformat() if hasattr(p.created_at, "isoformat") else str(p.created_at),
         "updated_at": p.updated_at.isoformat() if hasattr(p.updated_at, "isoformat") else str(p.updated_at)}
        for p in projects
    ]
    return {"projects": items, "meta": {"total": len(items), "skip": skip, "limit": limit}}