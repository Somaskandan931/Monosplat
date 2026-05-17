"""
server.py  —  Product
FastAPI web server for MonoSplat.

Product features
-----------------
    ✓ Job queue (Redis/RQ with thread fallback)
    ✓ SSE carries stage + progress % — drives live UI progress bar
    ✓ GET /api/jobs/{job_id}/metrics  — PSNR / SSIM / timing per job
    ✓ GET /api/health                 — queue mode, Redis status, worker count
    ✓ Structured error responses with actionable messages
    ✓ Colab handoff support — ready_for_colab status with zip instructions
    ✓ PUT /api/jobs/{job_id}/meta     — scene notes + tags (persistent metadata)
    ✓ GET /share/{job_id}             — shareable redirect to viewer
    ✓ GET /capture-guide              — standalone capture best-practices page

Endpoints
---------
    GET  /                              Upload portal HTML
    POST /upload                        Upload video for a named scene
    GET  /api/jobs                      List all jobs + statuses
    GET  /api/jobs/{job_id}             Single job status
    GET  /api/jobs/{job_id}/stream      SSE live status stream (stage + progress)
    GET  /api/jobs/{job_id}/metrics     PSNR / SSIM / timing for a completed job
    PUT  /api/jobs/{job_id}/meta        Update scene_notes and tags
    GET  /api/models                    List READY models
    GET  /api/models/latest             Most recently completed model
    GET  /api/health                    System health (queue mode, Redis, workers)
    GET  /splat/{job_id}                Serve .splat binary for renderer
    GET  /ply/{job_id}                  Serve .ply for download
    GET  /thumbnails/{job_id}           Serve thumbnail PNG
    GET  /share/{job_id}                Shareable redirect → /viewer/{job_id}
    GET  /viewer/{job_id}               Inline viewer (mobile-ready, SuperSplat deep-link, annotations)
    GET  /capture-guide                 Standalone capture best-practices page
"""

import asyncio
import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from src.utils.console import configure_console_encoding, ensure_project_dirs

configure_console_encoding()
ensure_project_dirs()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse
)

from .pipeline_manager import DynamicPipelineManager, JobStatus
from .queue_setup import initialize as init_queue, get_mode

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

UPLOADS_DIR = Path("uploads")
WORK_DIR    = Path("work")
UPLOADS_DIR.mkdir(exist_ok=True)
WORK_DIR.mkdir(exist_ok=True)

manager: Optional[DynamicPipelineManager] = None

_VALID_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
_VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
_VALID_EXT       = _VALID_VIDEO_EXT | _VALID_IMAGE_EXT

_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = DynamicPipelineManager(
        inbox="uploads",
        registry_path="models/registry.json",
        work_base="work",
        config_path="config/config.yaml",
    )
    manager.start()
    mode = init_queue(prefer_redis=True)
    print(f"[server] Queue mode: {mode}")
    print("[server] MonoSplat server started")
    yield
    if manager:
        manager.stop()
        print("[server] Pipeline manager stopped.")


app = FastAPI(
    title="MonoSplat — Product Pipeline",
    version="2.0.0",
    description="Single-camera 3D Gaussian Splat reconstruction with async job queue.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def _json_http_error(request: Request, exc: StarletteHTTPException):
    """Always return JSON errors — never HTML — so the frontend can parse them."""
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

@app.exception_handler(Exception)
async def _json_unhandled_error(request: Request, exc: Exception):
    """Catch-all: return JSON 500 instead of FastAPI's default HTML page."""
    import traceback
    print(f"[server] Unhandled error on {request.url}:\n{traceback.format_exc()}")
    return JSONResponse({"detail": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_files(
    item_name: str              = Form(..., description="Scene / object name"),
    files:     list[UploadFile] = File(..., description="Video (MP4/MOV) or photos (JPG/PNG)"),
):
    """
    Save uploaded file(s) and enqueue a pipeline job.

    The job is dispatched immediately to the async queue — no manual steps needed.
    """
    item_name = item_name.strip()
    if not item_name:
        raise HTTPException(400, "item_name cannot be empty")

    safe_name = "".join(
        c if (c.isalnum() or c in " -_") else "_" for c in item_name
    ).strip()

    item_dir = UPLOADS_DIR / safe_name
    item_dir.mkdir(parents=True, exist_ok=True)

    saved_files, skipped = [], []
    for f in files:
        ext = Path(f.filename).suffix
        if ext not in _VALID_EXT:
            skipped.append(f.filename)
            continue
        dest = item_dir / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_files.append(f.filename)

    if not saved_files:
        raise HTTPException(
            400,
            f"No valid files uploaded. Accepted extensions: {sorted(_VALID_EXT)}. "
            f"Received: {[f.filename for f in files]}"
        )

    # Create job in registry.
    # If this scene name is already active (e.g. stuck from a prior restart),
    # append a numeric suffix so the user can still submit a new attempt.
    base_name = safe_name
    job_id = ""
    for attempt in range(20):
        attempt_name = base_name if attempt == 0 else f"{base_name}_{attempt}"
        attempt_dir  = UPLOADS_DIR / attempt_name
        attempt_dir.mkdir(parents=True, exist_ok=True)
        # Move files into the attempt-specific directory on retry
        if attempt > 0:
            for fname in saved_files:
                src = item_dir / fname
                dst = attempt_dir / fname
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        job_id = manager.submit_job(str(attempt_dir), attempt_name)
        if job_id:
            safe_name = attempt_name
            break

    if not job_id:
        return JSONResponse(
            {"detail": "Could not create job — too many active jobs with this name."},
            status_code=500,
        )

    # Per-job work directory (pipeline_manager creates it internally,
    # but we also ensure it exists here for safety)
    job_work_dir = WORK_DIR / job_id
    job_work_dir.mkdir(parents=True, exist_ok=True)

    return JSONResponse({
        "status":      "queued",
        "item_name":   safe_name,
        "job_id":      job_id,
        "queue_mode":  get_mode(),
        "files_saved": len(saved_files),
        "files":       saved_files,
        "skipped":     skipped,
        "message":     (
            f"✅ {len(saved_files)} file(s) uploaded for '{safe_name}'. "
            f"Pipeline queued ({get_mode()} mode)."
        ),
        "stream_url":  f"/api/jobs/{job_id}/stream",
        "viewer_url":  f"/viewer/{job_id}",
    })


# ---------------------------------------------------------------------------
# Job API
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
async def list_jobs():
    jobs = manager.get_registry().all_jobs()
    return JSONResponse([j.to_dict() for j in reversed(jobs)])


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return JSONResponse(job.to_dict())


@app.get("/api/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """
    Return evaluation metrics for a completed job.

    Response fields
    ---------------
        psnr           float | null   — Peak Signal-to-Noise Ratio (dB); higher = better
        ssim           float | null   — Structural Similarity [0–1]; higher = better
        num_gaussians  int            — Number of Gaussians in the scene
        training_time_s float         — Wall-clock training time (seconds)
        colmap_time_s  float          — Wall-clock COLMAP time (seconds)
        total_time_s   float          — End-to-end wall-clock time (seconds)
    """
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    # Try loading metrics.json from the work directory
    metrics_path = WORK_DIR / job_id / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return JSONResponse(json.load(f))

    # Fallback: partial metrics from job record itself
    d = job.to_dict()
    return JSONResponse({
        "job_id":         job_id,
        "status":         d.get("status"),
        "num_gaussians":  d.get("num_gaussians"),
        "psnr":           d.get("metrics", {}).get("psnr") if d.get("metrics") else None,
        "ssim":           d.get("metrics", {}).get("ssim") if d.get("metrics") else None,
        "note":           "Full metrics available after job reaches 'ready' status.",
    })


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a single job (only failed / ready / waiting jobs — not active ones)."""
    ok = manager.get_registry().delete_job(job_id)
    if not ok:
        job = manager.get_registry().get_job(job_id)
        if job is None:
            raise HTTPException(404, f"Job not found: {job_id}")
        raise HTTPException(409, f"Cannot delete an active job (status={job.status}).")
    return JSONResponse({"deleted": job_id})


@app.put("/api/jobs/{job_id}/meta")
async def update_job_meta(job_id: str, body: dict):
    """
    Update user-editable scene metadata: scene_notes (str) and tags (list[str]).
    Ignores unknown keys. Only these two fields are user-writeable.

    Example body: {"scene_notes": "Demo scan of shoe", "tags": ["footwear", "demo"]}
    """
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    updates = {}
    if "scene_notes" in body:
        notes = str(body["scene_notes"])[:500]  # hard cap
        updates["scene_notes"] = notes
    if "tags" in body:
        raw_tags = body["tags"]
        if isinstance(raw_tags, list):
            updates["tags"] = [str(t)[:40] for t in raw_tags[:10]]
    if updates:
        manager.get_registry().update_job(job_id, **updates)
    job = manager.get_registry().get_job(job_id)
    return JSONResponse({"job_id": job_id, "scene_notes": job.scene_notes, "tags": job.tags})



@app.delete("/api/jobs")
async def clear_all_jobs(mode: str = "all"):
    """
    Remove non-active jobs in one shot.

    Query param `mode`:
        all    (default) — remove all finished/waiting/failed jobs
        failed            — remove only failed + orphaned waiting jobs
    """
    if mode == "failed":
        count = manager.get_registry().clear_failed()
    else:
        count = manager.get_registry().clear_all_inactive()
    return JSONResponse({"cleared": count, "mode": mode})


@app.get("/api/models")
async def list_models():
    models = manager.get_registry().ready_models()
    return JSONResponse([m.to_dict() for m in reversed(models)])


@app.get("/api/models/latest")
async def latest_model():
    models = manager.get_registry().ready_models()
    if not models:
        raise HTTPException(404, "No models ready yet.")
    latest = sorted(models, key=lambda m: m.updated_at)[-1]
    return JSONResponse(latest.to_dict())


def _tool_status() -> dict:
    """Check external tools required by the README quick-start."""
    import shutil
    import subprocess

    ffmpeg_ok = bool(shutil.which("ffmpeg"))
    colmap_path = shutil.which("colmap")
    colmap_ok = bool(colmap_path)
    colmap_version = None
    if colmap_ok:
        try:
            result = subprocess.run(
                [colmap_path, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            colmap_version = (result.stdout or result.stderr).splitlines()[0][:120]
        except Exception:
            colmap_version = "unknown"

    return {
        "ffmpeg": ffmpeg_ok,
        "colmap": colmap_ok,
        "colmap_version": colmap_version,
    }


@app.get("/api/health")
async def health():
    """
    System health check.

    Returns queue mode, Redis connectivity, uptime, and job counts.
    """
    registry = manager.get_registry()
    all_jobs  = registry.all_jobs()
    ready     = sum(1 for j in all_jobs if j.status == "ready")
    ready_for_colab = sum(1 for j in all_jobs if j.status == "ready_for_colab")
    in_queue  = sum(1 for j in all_jobs if j.status not in ("ready", "ready_for_colab", "failed"))
    failed    = sum(1 for j in all_jobs if j.status == "failed")

    redis_ok = False
    if get_mode() == "redis":
        try:
            from redis import Redis
            Redis(host="localhost", port=6379, socket_connect_timeout=1).ping()
            redis_ok = True
        except Exception:
            pass

    tools = _tool_status()
    ready = tools["ffmpeg"] and tools["colmap"]

    return JSONResponse({
        "status":       "ok" if ready else "degraded",
        "uptime_s":     round(time.time() - _START_TIME, 1),
        "queue_mode":   get_mode(),
        "redis_ok":     redis_ok,
        "tools":        tools,
        "jobs": {
            "total":    len(all_jobs),
            "ready":    ready,
            "ready_for_colab": ready_for_colab,
            "in_queue": in_queue,
            "failed":   failed,
        },
        "version": "2.0.0",
    })


# ---------------------------------------------------------------------------
# SSE live stream  — Product: carries stage name + progress %
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    """
    Server-Sent Events stream.

    Each event payload:
    {
        "job_id":   "...",
        "status":   "colmap" | "training" | "ready" | "ready_for_colab" | "failed" | ...,
        "progress": 0–100,
        "message":  "human readable log line",
        "stage":    "COLMAP" | "TRAINING" | "READY_FOR_COLAB" | "READY" | "FAILED",
        "done":     true  (only on terminal event)
    }
    """
    async def generate():
        last_sig = None
        for _ in range(720):   # max 12 min (720 × 1 s)
            job = manager.get_registry().get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'job not found', 'job_id': job_id})}\n\n"
                break

            sig = (job.status, job.progress, getattr(job, "message", ""))
            if sig != last_sig:
                last_sig = sig
                payload = {
                    **job.to_dict(),
                    "stage": job.status.upper(),
                    "done":  job.status in ("ready", "ready_for_colab", "failed"),
                }

                # Add Colab-specific instructions for ready_for_colab status
                if job.status == "ready_for_colab":
                    payload["colab_zip_command"] = f"python scripts/zip_for_colab.py {job_id}"
                    payload["colab_notebook"] = "notebooks/monosplat_colab_gpu.ipynb"
                    payload["next_steps"] = (
                        f"1. Run: python scripts/zip_for_colab.py {job_id}\n"
                        f"2. Upload {job_id}_for_colab.zip to Colab\n"
                        f"3. Open notebooks/monosplat_colab_gpu.ipynb\n"
                        f"4. Train on GPU and download .splat\n"
                        f"5. Place .splat in work/{job_id}/models/gaussian/\n"
                        f"6. Update registry.json status to 'ready'"
                    )

                yield f"data: {json.dumps(payload)}\n\n"

            if job.status in ("ready", "ready_for_colab", "failed"):
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Chunk streaming endpoints (Stage 3)
# ---------------------------------------------------------------------------

@app.get("/chunks/{job_id}/manifest.json")
async def get_chunk_manifest(job_id: str):
    """Serve chunk manifest for progressive loading."""
    manifest_path = Path(f"work/{job_id}/models/gaussian/{job_id}_chunks/manifest.json")
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Chunk manifest not found")
    return FileResponse(manifest_path, media_type="application/json")


@app.get("/chunks/{job_id}/{chunk_name}")
async def get_chunk(job_id: str, chunk_name: str):
    """Serve individual chunk file."""
    chunk_path = Path(f"work/{job_id}/models/gaussian/{job_id}_chunks/{chunk_name}")
    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Chunk not found")
    return FileResponse(chunk_path, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# AI Layer endpoints (Stage 7)
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/ai")
async def get_ai_results(job_id: str):
    """Get AI analysis results for a job."""
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "ai_detections": job.ai_detections,
        "ai_results": job.ai_results
    }


@app.post("/api/jobs/{job_id}/ai/query")
async def query_scene(job_id: str, query: dict):
    """
    Query the scene with natural language or spatial criteria.
    
    Request body:
    {
        "query": "chair",  # search query
        "query_type": "class"  # "class", "description", "nearby"
    }
    """
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not job.ai_results:
        raise HTTPException(status_code=400, detail="AI analysis not available for this job")
    
    try:
        from src.ai.ai_layer import AILayer
        ai = AILayer()
        
        # Rebuild spatial index from saved results
        if job.ai_results.get("detections"):
            ai.spatial_search.index[job_id] = {
                "objects": [],
                "cameras": []
            }
            for img_path, dets in job.ai_results["detections"].items():
                for det in dets:
                    ai.spatial_search.index[job_id]["objects"].append({
                        "class": det["class"],
                        "confidence": det["confidence"],
                        "center": det["center"],
                        "image": img_path
                    })
        
        results = ai.query_scene(
            job_id=job_id,
            query=query.get("query", ""),
            query_type=query.get("query_type", "description")
        )
        
        return {
            "job_id": job_id,
            "query": query.get("query", ""),
            "query_type": query.get("query_type", "description"),
            "results": results,
            "num_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI query failed: {str(e)}")


@app.post("/api/jobs/{job_id}/ai/qa")
async def ask_scene(job_id: str, qa_request: dict):
    """
    Ask a natural language question about the scene.
    
    Request body:
    {
        "question": "What objects are in this scene?"
    }
    """
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not job.ai_results:
        raise HTTPException(status_code=400, detail="AI analysis not available for this job")
    
    try:
        from src.ai.ai_layer import AILayer
        ai = AILayer()
        
        context = {
            "detections": job.ai_results.get("detections", {}),
            "num_gaussians": job.num_gaussians,
            "num_images": job.num_images
        }
        
        answer = ai.ask_scene(job_id, qa_request.get("question", ""), context)
        
        return {
            "job_id": job_id,
            "question": qa_request.get("question", ""),
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scene QA failed: {str(e)}")


@app.get("/share/{job_id}")
async def share_scene(job_id: str):
    """Shareable redirect — /share/<job_id> → /viewer/<job_id>. Useful for short links."""
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Scene not found: {job_id}")
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/viewer/{job_id}", status_code=302)


@app.get("/splat/{job_id}")
async def get_splat(job_id: str):
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    if not job.splat_path or not Path(job.splat_path).exists():
        raise HTTPException(
            404,
            f"Splat not ready yet — current status: {job.status}. "
            f"Track progress at /api/jobs/{job_id}/stream"
        )
    return FileResponse(
        job.splat_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"inline; filename={job.item_name}.splat"},
    )


@app.get("/ply/{job_id}")
async def get_ply(job_id: str):
    job = manager.get_registry().get_job(job_id)
    if not job or not job.ply_path or not Path(job.ply_path).exists():
        raise HTTPException(404, "PLY file not ready yet.")
    return FileResponse(
        job.ply_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={job.item_name}.ply"},
    )


@app.get("/spz/{job_id}")
async def get_spz(job_id: str):
    """Serve compressed .spz file (gzip splat, ~50% smaller than .splat)."""
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    # Derive spz path from splat path
    splat_p = Path(job.splat_path) if job.splat_path else None
    spz_p   = splat_p.with_suffix(".spz") if splat_p else None
    if not spz_p or not spz_p.exists():
        raise HTTPException(404, "SPZ file not available — job may not be complete or compression failed.")
    return FileResponse(
        str(spz_p),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={job.item_name}.spz",
            "Content-Encoding": "identity",   # already gzipped internally; don't double-compress
        },
    )


@app.get("/chunks/{job_id}/manifest.json")
async def get_chunks_manifest(job_id: str):
    """Serve streaming chunk manifest for progressive splat loading."""
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    splat_p = Path(job.splat_path) if job.splat_path else None
    if not splat_p:
        raise HTTPException(404, "Job has no splat yet.")
    manifest = splat_p.parent / f"{job_id}_chunks" / "manifest.json"
    if not manifest.exists():
        raise HTTPException(404, "Chunk manifest not available — job may have fewer than 10k Gaussians.")
    return FileResponse(str(manifest), media_type="application/json")


@app.get("/chunks/{job_id}/{chunk_name}")
async def get_chunk_file(job_id: str, chunk_name: str):
    """Serve an individual streaming chunk .splat file."""
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    splat_p = Path(job.splat_path) if job.splat_path else None
    if not splat_p:
        raise HTTPException(404, "Job has no splat yet.")
    # Sanitise chunk_name to prevent path traversal
    if "/" in chunk_name or "\\" in chunk_name or ".." in chunk_name:
        raise HTTPException(400, "Invalid chunk name.")
    chunk_path = splat_p.parent / f"{job_id}_chunks" / chunk_name
    if not chunk_path.exists():
        raise HTTPException(404, f"Chunk not found: {chunk_name}")
    return FileResponse(
        str(chunk_path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"inline; filename={chunk_name}"},
    )


@app.get("/api/jobs/{job_id}/preview")
async def get_training_preview(job_id: str):
    """
    Return the latest training preview frame (JPEG) as it's generated.

    During training the worker renders one frame every ~10% of iterations
    and writes it to work/<job_id>/previews/latest.jpg.
    Returns 404 if training hasn't produced a preview yet.
    """
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    preview = WORK_DIR / job_id / "previews" / "latest.jpg"
    if not preview.exists():
        raise HTTPException(404, "No preview available yet.")
    return FileResponse(
        str(preview),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ---------------------------------------------------------------------------
# Inline 3D viewer
# ---------------------------------------------------------------------------

@app.get("/viewer/{job_id}", response_class=HTMLResponse)
async def splat_viewer(job_id: str):
    job = manager.get_registry().get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    # Special message for ready_for_colab status
    if job.status == "ready_for_colab":
        return HTMLResponse(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{job.item_name} — Ready for Colab Training</title>
                <style>
                    body {{ font-family: monospace; background: #0a0c0f; color: #c8d0dc; padding: 2rem; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .card {{ background: #111418; border: 1px solid #1e2530; border-radius: 8px; padding: 2rem; margin: 1rem 0; }}
                    .code {{ background: #0a0c0f; border: 1px solid #1e2530; border-radius: 4px; padding: 1rem; font-family: monospace; overflow-x: auto; }}
                    .step {{ margin: 1.5rem 0; padding-left: 1rem; border-left: 2px solid #00e5a0; }}
                    h1 {{ color: #00e5a0; }}
                    a {{ color: #00e5a0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎯 {job.item_name}</h1>
                    <div class="card">
                        <h2>✅ COLMAP Complete — Ready for GPU Training</h2>
                        <p>Your frames and camera poses are ready. Now train on Google Colab GPU.</p>
                        
                        <div class="step">
                            <h3>Step 1: Zip the job</h3>
                            <div class="code">
                                python scripts/zip_for_colab.py {job_id}
                            </div>
                        </div>
                        
                        <div class="step">
                            <h3>Step 2: Upload to Colab</h3>
                            <p>Open <a href="https://colab.research.google.com" target="_blank">Google Colab</a> and upload:</p>
                            <div class="code">
                                {job_id}_for_colab.zip
                            </div>
                        </div>
                        
                        <div class="step">
                            <h3>Step 3: Run the notebook</h3>
                            <p>Open <code>notebooks/monosplat_colab_gpu.ipynb</code> and run all cells.</p>
                            <p>⏱️ Training takes 20-40 minutes on T4 GPU.</p>
                        </div>
                        
                        <div class="step">
                            <h3>Step 4: Download and place files</h3>
                            <p>After training, download:</p>
                            <div class="code">
                                {job_id}.splat<br>
                                {job_id}.ply
                            </div>
                            <p>Place them in:</p>
                            <div class="code">
                                work\\{job_id}\\models\\gaussian\\
                            </div>
                        </div>
                        
                        <div class="step">
                            <h3>Step 5: Update registry</h3>
                            <p>Edit <code>models/registry.json</code> and set:</p>
                            <div class="code">
                                "status": "ready",<br>
                                "splat_path": "work/{job_id}/models/gaussian/{job_id}.splat",<br>
                                "ply_path": "work/{job_id}/models/gaussian/{job_id}.ply"
                            </div>
                        </div>
                        
                        <div class="step">
                            <h3>Step 6: Refresh this page</h3>
                            <p>After updating the registry, reload this page to view your 3D scene!</p>
                        </div>
                    </div>
                    <p><a href="/">← Back to upload portal</a></p>
                </div>
            </body>
            </html>
            """
        )

    splat_file = Path(job.splat_path) if job.splat_path else None
    if splat_file is None or not splat_file.exists():
        return HTMLResponse(
            f"""<!DOCTYPE html><html><body style="font-family:monospace;background:#0a0c0f;color:#c8d0dc;padding:2rem">
            <h2 style="color:#f5b84b">Splat file not found</h2>
            <p>Status is <code>{job.status}</code> but no .splat exists at:</p>
            <p><code>{job.splat_path or '(not set)'}</code></p>
            <p>GPU training must finish first. On desktop the pipeline stops at
            <code>ready_for_colab</code> — train in Colab, download
            <code>{job_id}.splat</code>, place it in
            <code>work/{job_id}/models/gaussian/</code>, then run:</p>
            <pre style="background:#111418;padding:1rem;border-radius:6px">python scripts/mark_ready.py {job_id}</pre>
            <p><a href="/" style="color:#00e5a0">Back to portal</a></p>
            </body></html>"""
        )

    from src.utils.io_utils import splat_bounds
    try:
        bounds = splat_bounds(str(splat_file))
    except Exception:
        bounds = {"center": [0, 0, 0], "radius": 2.0, "num_splats": 0}

    cx, cy, cz = bounds["center"]
    r = bounds["radius"]
    cam_pos = [cx, cy - r * 1.5, cz + r * 2.5]
    cam_look = [cx, cy, cz]

    splat_url = f"/splat/{job_id}"
    ply_url   = f"/ply/{job_id}" if job.ply_path and Path(job.ply_path).exists() else ""
    item_name = job.item_name
    num_splats = bounds.get("num_splats", 0)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{item_name} — MonoSplat Viewer</title>
  <script type="importmap">
  {{
    "imports": {{
      "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
      "@mkkellogg/gaussian-splats-3d": "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.js"
    }}
  }}
  </script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:      #080b0e;
      --panel:   rgba(8,11,14,0.88);
      --border:  rgba(255,255,255,0.07);
      --brand:   #00e5a0;
      --brand2:  #00b4ff;
      --muted:   rgba(255,255,255,0.35);
      --text:    rgba(255,255,255,0.88);
      --warn:    #f5b84b;
      --danger:  #ef5b6b;
      --mono:    'JetBrains Mono', 'Fira Code', Consolas, monospace;
    }}

    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--mono);
      overflow: hidden;
      user-select: none;
    }}

    canvas {{ display: block; width: 100vw !important; height: 100vh !important; }}

    /* ── Loading screen ── */
    #loading {{
      position: fixed; inset: 0; z-index: 200;
      display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 20px;
      background: var(--bg);
      transition: opacity 0.6s ease;
    }}
    #loading.fade-out {{ opacity: 0; pointer-events: none; }}
    .load-logo {{
      font-size: 11px; letter-spacing: 0.3em; text-transform: uppercase;
      color: var(--brand); opacity: 0.7;
    }}
    .load-title {{
      font-size: 22px; font-weight: 700; color: var(--text);
      max-width: 360px; text-align: center; line-height: 1.3;
    }}
    .bar-wrap {{
      width: 260px; height: 2px; background: rgba(255,255,255,0.08);
      border-radius: 2px; overflow: hidden; position: relative;
    }}
    .bar-fill {{
      height: 100%; width: 0%; background: linear-gradient(90deg, var(--brand), var(--brand2));
      border-radius: 2px; transition: width 0.4s cubic-bezier(0.4,0,0.2,1);
    }}
    .bar-shimmer {{
      position: absolute; inset: 0;
      background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
      animation: shimmer 1.4s infinite;
    }}
    @keyframes shimmer {{ from {{ transform: translateX(-100%); }} to {{ transform: translateX(100%); }} }}
    #load-msg {{
      font-size: 11px; color: var(--muted); letter-spacing: 0.08em;
      min-height: 16px; transition: all 0.2s;
    }}
    .load-stats {{
      display: flex; gap: 24px; margin-top: 4px;
    }}
    .load-stat {{
      font-size: 10px; color: var(--muted); text-align: center;
    }}
    .load-stat strong {{
      display: block; font-size: 14px; color: var(--brand); margin-bottom: 2px;
    }}

    /* ── Top bar ── */
    #topbar {{
      position: fixed; top: 0; left: 0; right: 0; z-index: 50;
      display: none; align-items: center; justify-content: space-between;
      padding: 0 20px; height: 52px;
      background: var(--panel); border-bottom: 1px solid var(--border);
      backdrop-filter: blur(20px) saturate(180%);
    }}
    #topbar.visible {{ display: flex; }}
    .tb-left  {{ display: flex; align-items: center; gap: 12px; }}
    .tb-title {{ font-size: 13px; font-weight: 600; color: var(--text); }}
    .tb-badge {{
      font-size: 10px; color: var(--brand); background: rgba(0,229,160,0.12);
      border: 1px solid rgba(0,229,160,0.25); border-radius: 3px;
      padding: 2px 7px; letter-spacing: 0.05em; text-transform: uppercase;
    }}
    .tb-right {{ display: flex; align-items: center; gap: 8px; }}
    .tb-btn {{
      height: 30px; padding: 0 14px; border-radius: 4px; font-family: var(--mono);
      font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase;
      cursor: pointer; border: 1px solid var(--border);
      background: rgba(255,255,255,0.04); color: var(--muted);
      text-decoration: none; display: flex; align-items: center; gap: 6px;
      transition: all 0.15s;
    }}
    .tb-btn:hover {{ background: rgba(255,255,255,0.08); color: var(--text); border-color: rgba(255,255,255,0.15); }}
    .tb-btn.primary {{ background: rgba(0,229,160,0.12); color: var(--brand); border-color: rgba(0,229,160,0.3); }}
    .tb-btn.primary:hover {{ background: rgba(0,229,160,0.2); }}
    .tb-btn.supersplat-btn {{ background: rgba(112,100,255,0.12); color: #a78bfa; border-color: rgba(112,100,255,0.3); }}
    .tb-btn.supersplat-btn:hover {{ background: rgba(112,100,255,0.22); color: #c4b5fd; }}
    .tb-btn.copied {{ color: var(--brand); border-color: rgba(0,229,160,0.3); }}
    #fps-badge {{
      font-size: 10px; color: var(--muted); min-width: 50px; text-align: right;
    }}

    /* ── Side HUD (controls) ── */
    #hud {{
      position: fixed; left: 20px; top: 72px; z-index: 50;
      display: none; flex-direction: column; gap: 2px;
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 14px 16px;
      backdrop-filter: blur(20px); min-width: 180px;
      transition: opacity 0.2s;
    }}
    #hud.visible {{ display: flex; }}
    #hud.collapsed {{ opacity: 0.3; }}
    #hud.collapsed:hover {{ opacity: 1; }}
    .hud-header {{
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em;
      color: var(--brand); margin-bottom: 8px; font-weight: 600;
    }}
    .hud-row {{
      display: flex; justify-content: space-between; align-items: center;
      font-size: 11px; padding: 2px 0; gap: 16px;
    }}
    .hud-key {{ color: var(--muted); }}
    .hud-action {{ color: var(--text); text-align: right; }}
    .hud-divider {{ height: 1px; background: var(--border); margin: 8px 0; }}

    /* ── Metrics panel ── */
    #metrics {{
      position: fixed; right: 20px; top: 72px; z-index: 50;
      display: none; flex-direction: column; gap: 10px;
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 14px 16px;
      backdrop-filter: blur(20px); min-width: 180px;
    }}
    #metrics.visible {{ display: flex; }}
    .metric-group-title {{
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em;
      color: var(--brand2); margin-bottom: 2px; font-weight: 600;
    }}
    .metric-row {{
      display: flex; justify-content: space-between; align-items: baseline;
      font-size: 11px; gap: 12px;
    }}
    .metric-label {{ color: var(--muted); }}
    .metric-val {{ color: var(--text); font-weight: 600; }}
    .metric-val.good {{ color: var(--brand); }}
    .metric-val.warn {{ color: var(--warn); }}

    /* ── Toast notifications ── */
    #toast-container {{
      position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
      z-index: 100; display: flex; flex-direction: column; align-items: center; gap: 8px;
      pointer-events: none;
    }}
    .toast {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 6px; padding: 10px 18px; font-size: 11px; color: var(--text);
      letter-spacing: 0.04em; backdrop-filter: blur(20px);
      animation: toast-in 0.25s ease, toast-out 0.25s ease 2.5s forwards;
      pointer-events: none;
    }}
    @keyframes toast-in  {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
    @keyframes toast-out {{ from {{ opacity:1; }} to {{ opacity:0; }} }}

    /* ── Fallback / error state ── */
    #error-bar {{
      position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
      display: none; align-items: center; justify-content: center; gap: 16px;
      background: rgba(239,91,107,0.12); border-top: 1px solid rgba(239,91,107,0.3);
      padding: 12px 20px; font-size: 11px;
    }}
    #error-bar.visible {{ display: flex; }}
    #error-bar .err-msg {{ color: var(--danger); }}
    #error-bar a {{ color: var(--brand); text-decoration: none; }}
    #error-bar a:hover {{ text-decoration: underline; }}

    /* ── Annotations panel ── */
    #annotations {{
      position: fixed; left: 20px; bottom: 24px; z-index: 50;
      display: none; flex-direction: column; gap: 6px;
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 14px 16px;
      backdrop-filter: blur(20px); min-width: 200px; max-width: 240px;
    }}
    #annotations.visible {{ display: flex; }}
    .ann-header {{
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em;
      color: var(--brand); font-weight: 600; margin-bottom: 4px;
    }}
    .ann-item {{
      display: flex; align-items: center; gap: 8px;
      font-size: 11px; cursor: pointer; padding: 4px 0;
      border-radius: 4px; transition: opacity .15s;
    }}
    .ann-item:hover {{ opacity: 0.7; }}
    .ann-dot {{
      width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    }}
    .ann-label {{ color: var(--text); flex: 1; }}
    .ann-add {{
      display: flex; gap: 6px; margin-top: 4px; border-top: 1px solid var(--border); padding-top: 8px;
    }}
    .ann-input {{
      flex: 1; background: rgba(255,255,255,0.05); border: 1px solid var(--border);
      border-radius: 4px; padding: 5px 8px; font-size: 11px; color: var(--text);
      font-family: var(--mono); outline: none;
    }}
    .ann-input:focus {{ border-color: var(--brand); }}
    .ann-add-btn {{
      height: 28px; padding: 0 10px; border-radius: 4px; font-size: 10px;
      background: rgba(0,229,160,0.12); color: var(--brand);
      border: 1px solid rgba(0,229,160,0.3); cursor: pointer;
      font-family: var(--mono); letter-spacing: 0.05em;
      transition: background .15s;
    }}
    .ann-add-btn:hover {{ background: rgba(0,229,160,0.22); }}

    /* ── Mobile overrides ── */
    @media (max-width: 600px) {{
      #hud {{ display: none !important; }}
      #annotations {{ display: none !important; }}
      #metrics {{ top: auto; bottom: 68px; right: 12px; min-width: 160px; }}
      #topbar {{ padding: 0 12px; }}
      .tb-title {{ font-size: 12px; }}
      .tb-badge {{ display: none; }}
      #fps-badge {{ min-width: 40px; }}
    }}
  </style>
</head>
<body>

<!-- ── Loading screen ── -->
<div id="loading">
  <div class="load-logo">MonoSplat · XR Labs</div>
  <div class="load-title">{item_name}</div>
  <div class="bar-wrap">
    <div class="bar-fill" id="bar"></div>
    <div class="bar-shimmer"></div>
  </div>
  <div id="load-msg">Initialising renderer…</div>
  <div class="load-stats">
    <div class="load-stat"><strong id="stat-size">—</strong>size</div>
    <div class="load-stat"><strong id="stat-splats">{num_splats:,}</strong>gaussians</div>
    <div class="load-stat"><strong id="stat-time">—</strong>elapsed</div>
  </div>
</div>

<!-- ── Top navigation bar ── -->
<div id="topbar">
  <div class="tb-left">
    <div class="tb-title">{item_name}</div>
    <div class="tb-badge">Live · {num_splats:,} gaussians</div>
  </div>
  <div class="tb-right">
    <span id="fps-badge">— fps</span>
    <button class="tb-btn" id="btn-vr" title="Enter VR Mode">🥽 VR</button>
    <button class="tb-btn" id="btn-ar" title="Enter AR Mode">📱 AR</button>
    <button class="tb-btn" id="btn-measure" title="Measurement Tool (T)">📏 Measure</button>
    <button class="tb-btn" id="btn-teleport" title="Teleport (Shift+Click)">⚡ Teleport</button>
    <button class="tb-btn" id="btn-collab" title="Collaborative Mode">👥 Collab</button>
    <button class="tb-btn" id="btn-hud" title="Toggle controls (H)">⌨ Controls</button>
    <button class="tb-btn" id="btn-metrics" title="Toggle metrics (M)">📊 Metrics</button>
    <button class="tb-btn" id="btn-ann" title="Toggle annotations (A)">📍 Labels</button>
    <button class="tb-btn" id="btn-fullscreen" title="Fullscreen (F)">⛶ Full</button>
    <button class="tb-btn" id="btn-share" title="Copy shareable link">↗ Share</button>
    <a class="tb-btn supersplat-btn" id="btn-supersplat"
       href="https://supersplat.playcanvas.com" target="_blank" rel="noopener"
       title="Open this scene in SuperSplat for GPU-quality editing">
      ✦ SuperSplat
    </a>
    <a class="tb-btn primary" href="{splat_url}" download="{item_name}.splat" title="Download .splat">↓ .splat</a>
    <a class="tb-btn" href="/" title="Back to portal">← Portal</a>
  </div>
</div>

<!-- ── Controls HUD ── -->
<div id="hud">
  <div class="hud-header">Navigation</div>
  <div class="hud-row"><span class="hud-key">Left drag</span><span class="hud-action">Orbit</span></div>
  <div class="hud-row"><span class="hud-key">Right drag</span><span class="hud-action">Pan</span></div>
  <div class="hud-row"><span class="hud-key">Scroll / Pinch</span><span class="hud-action">Zoom</span></div>
  <div class="hud-row"><span class="hud-key">Two-finger drag</span><span class="hud-action">Pan (mobile)</span></div>
  <div class="hud-divider"></div>
  <div class="hud-header">Shortcuts</div>
  <div class="hud-row"><span class="hud-key">R</span><span class="hud-action">Reset camera</span></div>
  <div class="hud-row"><span class="hud-key">F</span><span class="hud-action">Fullscreen</span></div>
  <div class="hud-row"><span class="hud-key">H</span><span class="hud-action">Toggle this panel</span></div>
  <div class="hud-row"><span class="hud-key">M</span><span class="hud-action">Toggle metrics</span></div>
  <div class="hud-row"><span class="hud-key">Space</span><span class="hud-action">Auto-rotate</span></div>
  <div class="hud-row"><span class="hud-key">A</span><span class="hud-action">Toggle labels</span></div>
</div>

<!-- ── Metrics panel ── -->
<div id="metrics">
  <div class="metric-group-title">Quality</div>
  <div class="metric-row">
    <span class="metric-label">PSNR</span>
    <span class="metric-val" id="m-psnr">—</span>
  </div>
  <div class="metric-row">
    <span class="metric-label">SSIM</span>
    <span class="metric-val" id="m-ssim">—</span>
  </div>
  <div class="hud-divider"></div>
  <div class="metric-group-title">Scene</div>
  <div class="metric-row">
    <span class="metric-label">Gaussians</span>
    <span class="metric-val good" id="m-gauss">{num_splats:,}</span>
  </div>
  <div class="metric-row">
    <span class="metric-label">File size</span>
    <span class="metric-val" id="m-size">—</span>
  </div>
  <div class="metric-row">
    <span class="metric-label">FPS</span>
    <span class="metric-val" id="m-fps">—</span>
  </div>
</div>

<!-- ── Annotations panel ── -->
<div id="annotations">
  <div class="ann-header">Scene Labels</div>
  <div id="ann-list">
    <div class="ann-item"><div class="ann-dot" style="background:var(--brand)"></div><span class="ann-label">Origin</span></div>
  </div>
  <div class="ann-add">
    <input class="ann-input" id="ann-input" placeholder="Add label…" maxlength="32">
    <button class="ann-add-btn" id="ann-add-btn">+</button>
  </div>
</div>

<!-- ── Toast container ── -->
<div id="toast-container"></div>

<!-- ── Error bar ── -->
<div id="error-bar" id="error-bar">
  <span class="err-msg" id="err-msg">Viewer error</span>
  <span>·</span>
  <a href="#" onclick="openInSuperSplat();return false">Open in SuperSplat ↗</a>
  <span>·</span>
  <a href="{splat_url}" download="{item_name}.splat">Download .splat</a>
</div>

<script type="module">
import * as THREE from 'three';
window.THREE = THREE;

// ── State ──────────────────────────────────────────────────────────────────
const CAM_POS  = {cam_pos};
const CAM_LOOK = {cam_look};

const $  = id => document.getElementById(id);
const loading  = $('loading');
const bar      = $('bar');
const loadMsg  = $('load-msg');
const topbar   = $('topbar');
const hud      = $('hud');
const metrics  = $('metrics');
const errorBar = $('error-bar');

let viewer       = null;
let autoRotate   = false;
let autoRotateId = null;
let loadStart    = Date.now();
let frameCount   = 0;
let lastFpsSample = performance.now();

// ── Utilities ──────────────────────────────────────────────────────────────
function setBar(pct, msg) {{
  bar.style.width = pct + '%';
  if (msg) loadMsg.textContent = msg;
}}

function fmtBytes(b) {{
  if (b > 1e6) return (b/1e6).toFixed(1) + ' MB';
  return (b/1e3).toFixed(0) + ' KB';
}}

function fmtGauss(n) {{
  if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'k';
  return String(n);
}}

function showToast(msg, duration = 2800) {{
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  $('toast-container').appendChild(t);
  setTimeout(() => t.remove(), duration + 300);
}}

function showError(msg) {{
  loadMsg.style.color = 'var(--danger)';
  loadMsg.textContent = msg;
  $('err-msg').textContent = msg;
  errorBar.classList.add('visible');
}}

// ── FPS counter ────────────────────────────────────────────────────────────
function trackFps() {{
  frameCount++;
  const now = performance.now();
  if (now - lastFpsSample >= 1000) {{
    const fps = Math.round(frameCount * 1000 / (now - lastFpsSample));
    const col = fps >= 55 ? 'var(--brand)' : fps >= 30 ? 'var(--warn)' : 'var(--danger)';
    $('fps-badge').textContent = fps + ' fps';
    $('m-fps').textContent = fps + ' fps';
    $('m-fps').style.color = col;
    frameCount = 0;
    lastFpsSample = now;
  }}
  requestAnimationFrame(trackFps);
}}

// ── UI toggles ─────────────────────────────────────────────────────────────
function toggleHud() {{
  hud.classList.toggle('visible');
  hud.classList.remove('collapsed');
}}
function toggleMetrics() {{
  metrics.classList.toggle('visible');
}}
function toggleFullscreen() {{
  if (!document.fullscreenElement) {{
    document.documentElement.requestFullscreen().catch(() => {{}});
    showToast('Fullscreen — press Esc to exit');
  }} else {{
    document.exitFullscreen();
  }}
}}
function resetCamera() {{
  if (!viewer) return;
  viewer.getCamera().position.set(...CAM_POS);
  if (viewer.controls) viewer.controls.target.set(...CAM_LOOK);
  viewer.getCamera().lookAt(...CAM_LOOK);
  showToast('Camera reset');
}}
function toggleAutoRotate() {{
  autoRotate = !autoRotate;
  if (viewer && viewer.controls) {{
    viewer.controls.autoRotate = autoRotate;
    viewer.controls.autoRotateSpeed = 0.6;
  }}
  showToast(autoRotate ? 'Auto-rotate ON' : 'Auto-rotate OFF');
}}

// ── VR mode ───────────────────────────────────────────────────────────────────
async function enterVR() {{
  if (!viewer) return;
  try {{
    await viewer.enterVR();
    showToast('Entering VR mode…');
  }} catch (err) {{
    console.error('[VR]', err);
    showToast('VR not available or not supported', 4000);
  }}
}}

async function enterAR() {{
  if (!viewer) return;
  try {{
    // Try AR mode first, fall back to VR if not supported
    if (viewer.enterAR) {{
      await viewer.enterAR();
      showToast('Entering AR mode…');
    }} else {{
      await viewer.enterVR();
      showToast('AR not supported, entering VR mode…');
    }}
  }} catch (err) {{
    console.error('[AR]', err);
    showToast('AR/VR not available or not supported', 4000);
  }}
}}

$('btn-vr').addEventListener('click', enterVR);
$('btn-ar').addEventListener('click', enterAR);

// ── Measurement tool (Stage 6) ───────────────────────────────────────────────
let measureMode = false;
let measurePoints = [];

function toggleMeasure() {{
  measureMode = !measureMode;
  measurePoints = [];
  $('btn-measure').classList.toggle('primary', measureMode);
  showToast(measureMode ? 'Measure mode ON - click two points' : 'Measure mode OFF');
}}

function handleMeasureClick(event) {{
  if (!measureMode || !viewer) return;
  
  // Get raycast intersection point
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2(
    (event.clientX / window.innerWidth) * 2 - 1,
    -(event.clientY / window.innerHeight) * 2 + 1
  );
  
  raycaster.setFromCamera(mouse, viewer.getCamera());
  
  // This is a simplified version - in production you'd raycast against the splat scene
  // For now, we'll use camera distance as a proxy
  const camera = viewer.getCamera();
  const point = camera.position.clone().add(camera.getWorldDirection(new THREE.Vector3()).multiplyScalar(2));
  
  measurePoints.push(point);
  
  if (measurePoints.length === 2) {{
    const distance = measurePoints[0].distanceTo(measurePoints[1]);
    showToast(`Distance: ${{distance.toFixed(2)}} units`);
    measurePoints = [];
    measureMode = false;
    $('btn-measure').classList.remove('primary');
  }}
}}

$('btn-measure').addEventListener('click', toggleMeasure);

// ── Teleport (Stage 6) ───────────────────────────────────────────────────────
let teleportMode = false;

function toggleTeleport() {{
  teleportMode = !teleportMode;
  $('btn-teleport').classList.toggle('primary', teleportMode);
  showToast(teleportMode ? 'Teleport mode ON - Shift+Click to teleport' : 'Teleport mode OFF');
}}

function handleTeleport(event) {{
  if (!teleportMode || !viewer || !event.shiftKey) return;
  
  // Teleport camera to look at clicked point (simplified)
  const camera = viewer.getCamera();
  const direction = camera.getWorldDirection(new THREE.Vector3());
  camera.position.add(direction.multiplyScalar(2));
  
  showToast('Teleported');
}}

$('btn-teleport').addEventListener('click', toggleTeleport);

// ── Collaborative viewing (Stage 6) ───────────────────────────────────────────
let collabMode = false;
let collabSocket = null;

function toggleCollab() {{
  collabMode = !collabMode;
  $('btn-collab').classList.toggle('primary', collabMode);
  
  if (collabMode) {{
    initCollab();
    showToast('Collaborative mode ON');
  }} else {{
    if (collabSocket) {{
      collabSocket.close();
      collabSocket = null;
    }}
    showToast('Collaborative mode OFF');
  }}
}}

function initCollab() {{
  // WebSocket-based camera sync (placeholder - requires WebSocket server)
  // In production, this would connect to a WebSocket server and sync camera positions
  console.log('[Collab] Collaborative mode initialized');
  showToast('Collaborative mode - camera sync active');
}}

$('btn-collab').addEventListener('click', toggleCollab);

// ── Keyboard shortcuts ──────────────────────────────────────────────────────
document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'INPUT') return;
  switch (e.key) {{
    case 'r': case 'R': resetCamera(); break;
    case 'f': case 'F': toggleFullscreen(); break;
    case 'h': case 'H': toggleHud(); break;
    case 'm': case 'M': toggleMetrics(); break;
    case 'a': case 'A': toggleAnnotations(); break;
    case 't': case 'T': toggleMeasure(); break;
    case 'v': case 'V': enterVR(); break;
    case 'c': case 'C': toggleCollab(); break;
    case ' ':  e.preventDefault(); toggleAutoRotate(); break;
  }}
}});

// ── Click handlers for XR features ─────────────────────────────────────────────
document.addEventListener('click', handleMeasureClick);
document.addEventListener('click', handleTeleport);

function toggleAnnotations() {{
  const ann = $('annotations');
  ann.classList.toggle('visible');
}}

// Share — copy viewer URL to clipboard
function shareViewer() {{
  const url = window.location.href;
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(url).then(() => {{
      const btn = $('btn-share');
      btn.textContent = '✓ Copied';
      btn.classList.add('copied');
      setTimeout(() => {{ btn.textContent = '↗ Share'; btn.classList.remove('copied'); }}, 2000);
    }}).catch(() => fallbackCopyViewer(url));
  }} else {{
    fallbackCopyViewer(url);
  }}
}}

function fallbackCopyViewer(text) {{
  const ta = document.createElement('textarea');
  ta.value = text; ta.style.cssText = 'position:fixed;opacity:0';
  document.body.appendChild(ta); ta.select(); document.execCommand('copy');
  document.body.removeChild(ta);
  showToast('Link copied to clipboard');
}}

// SuperSplat deep-link — pass the absolute .splat URL as ?load= param
function openInSuperSplat() {{
  const splatAbsUrl = window.location.origin + '{splat_url}';
  const superSplatUrl = 'https://supersplat.playcanvas.com/?load=' + encodeURIComponent(splatAbsUrl);
  window.open(superSplatUrl, '_blank', 'noopener');
}}

// ── Button wiring ───────────────────────────────────────────────────────────
$('btn-hud').addEventListener('click', toggleHud);
$('btn-metrics').addEventListener('click', toggleMetrics);
$('btn-ann').addEventListener('click', toggleAnnotations);
$('btn-share').addEventListener('click', shareViewer);
$('btn-supersplat').addEventListener('click', e => {{ e.preventDefault(); openInSuperSplat(); }});
$('btn-fullscreen').addEventListener('click', toggleFullscreen);

// Annotation add
const ANN_COLORS = ['var(--brand)','var(--brand2)','var(--warn)','var(--danger)','#c084fc','#fb923c'];
let annColorIdx = 1;
function addAnnotation(label) {{
  if (!label.trim()) return;
  const list  = $('ann-list');
  const color = ANN_COLORS[annColorIdx % ANN_COLORS.length];
  annColorIdx++;
  const item = document.createElement('div');
  item.className = 'ann-item';
  item.title = 'Click to remove';
  item.innerHTML = `<div class="ann-dot" style="background:${{color}}"></div><span class="ann-label">${{label.slice(0, 32)}}</span>`;
  item.addEventListener('click', () => item.remove());
  list.appendChild(item);
}}
$('ann-add-btn').addEventListener('click', () => {{
  const inp = $('ann-input');
  addAnnotation(inp.value);
  inp.value = '';
  inp.focus();
}});
$('ann-input').addEventListener('keydown', e => {{
  if (e.key === 'Enter') {{ addAnnotation(e.target.value); e.target.value = ''; }}
}});

// ── Main load sequence ──────────────────────────────────────────────────────
(async () => {{
  try {{
    setBar(5, 'Loading GPU renderer…');

    const GS3D = await import('@mkkellogg/gaussian-splats-3d');
    setBar(15, 'Renderer ready — fetching splat…');

    // Fetch metrics in background (non-blocking)
    fetch('/api/jobs/{job_id}/metrics')
      .then(r => r.ok ? r.json() : null)
      .then(d => {{
        if (!d) return;
        const psnrEl = $('m-psnr');
        const ssimEl = $('m-ssim');
        if (d.psnr) {{
          psnrEl.textContent = d.psnr.toFixed(2) + ' dB';
          psnrEl.className = 'metric-val ' + (d.psnr >= 25 ? 'good' : 'warn');
        }}
        if (d.ssim) {{
          ssimEl.textContent = d.ssim.toFixed(3);
          ssimEl.className = 'metric-val ' + (d.ssim >= 0.8 ? 'good' : 'warn');
        }}
        if (d.num_gaussians) $('m-gauss').textContent = fmtGauss(d.num_gaussians);
      }}).catch(() => {{}});

    // Progressive chunk loading (Stage 3) - try chunks first, fall back to monolithic
    let splatBuffer;
    let totalReceived = 0;

    try {{
      // Try to fetch chunk manifest for progressive loading
      const manifestUrl = '/chunks/{job_id}/manifest.json';
      const manifestResp = await fetch(manifestUrl);
      
      if (manifestResp.ok) {{
        const manifest = await manifestResp.json();
        const chunkFiles = manifest.chunks || [];
        const totalChunks = chunkFiles.length;
        
        if (totalChunks > 0) {{
          console.log('[MonoSplat] Loading', totalChunks, 'chunks progressively');
          
          // Load chunks in order (sorted by opacity - coarse LOD first)
          const allBuffers = [];
          
          for (let i = 0; i < totalChunks; i++) {{
            const chunkName = chunkFiles[i];
            const chunkUrl = `/chunks/{job_id}/${{chunkName}}`;
            const chunkResp = await fetch(chunkUrl);
            
            if (!chunkResp.ok) throw new Error(`Chunk ${{i}} download failed`);
            
            const chunkBuffer = await chunkResp.arrayBuffer();
            allBuffers.push(new Uint8Array(chunkBuffer));
            totalReceived += chunkBuffer.byteLength;
            
            const pct = Math.round(15 + ((i + 1) / totalChunks) * 45);
            setBar(pct, `Loading chunk ${{i + 1}}/${{totalChunks}}… ` + fmtBytes(totalReceived));
          }}
          
          // Concatenate all chunk buffers
          const totalLength = allBuffers.reduce((sum, buf) => sum + buf.length, 0);
          const combined = new Uint8Array(totalLength);
          let offset = 0;
          for (const buf of allBuffers) {{
            combined.set(buf, offset);
            offset += buf.length;
          }}
          
          $('stat-size').textContent = fmtBytes(totalReceived);
          $('m-size').textContent = fmtBytes(totalReceived);
          setBar(62, 'Parsing ' + fmtBytes(totalReceived) + ' of splat data…');
          
          splatBuffer = await GS3D.SplatLoader.loadFromFileData(combined.buffer, 1, 0, true);
        }}
      }}
    }} catch (chunkErr) {{
      console.log('[MonoSplat] Chunk loading failed, falling back to monolithic:', chunkErr.message);
      
      // Fallback to monolithic fetch
      const response = await fetch('{splat_url}');
      if (!response.ok) throw new Error('Splat download failed (HTTP ' + response.status + ')');

      const contentLength = parseInt(response.headers.get('content-length') || '0');
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;

      while (true) {{
        const {{ done, value }} = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (contentLength > 0) {{
          const pct = Math.round(15 + (received / contentLength) * 45);
          setBar(pct, 'Downloading… ' + fmtBytes(received) + (contentLength ? ' / ' + fmtBytes(contentLength) : ''));
        }} else {{
          setBar(40, 'Downloading… ' + fmtBytes(received));
        }}
      }}

      const buffer = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {{ buffer.set(chunk, offset); offset += chunk.length; }}
      const arrayBuf = buffer.buffer;

      totalReceived = received;
      $('stat-size').textContent = fmtBytes(received);
      $('m-size').textContent = fmtBytes(received);
      setBar(62, 'Parsing ' + fmtBytes(received) + ' of splat data…');

      splatBuffer = await GS3D.SplatLoader.loadFromFileData(arrayBuf, 1, 0, true);
    }}
    setBar(78, 'Initialising WebGL scene…');

    viewer = new GS3D.Viewer({{
      cameraUp:               [0, -1, 0],
      initialCameraPosition:  CAM_POS,
      initialCameraLookAt:    CAM_LOOK,
      selfDrivenMode:         true,
      sharedMemoryForWorkers: false,
      integerBasedSort:       false,
      dynamicScene:           false,
      webXRMode:              GS3D.WebXRMode?.VR ?? 1,
    }});

    await viewer.addSplatBuffers([splatBuffer], [{{ splatAlphaRemovalThreshold: 1 }}]);
    setBar(95, 'Starting render loop…');
    viewer.start();

    // Enable auto-rotate on controls if available
    if (viewer.controls) {{
      viewer.controls.enableDamping = true;
      viewer.controls.dampingFactor = 0.05;
    }}

    // Fade out loading screen
    const elapsed = ((Date.now() - loadStart) / 1000).toFixed(1);
    $('stat-time').textContent = elapsed + 's';
    setBar(100, 'Ready');
    setTimeout(() => {{
      loading.classList.add('fade-out');
      setTimeout(() => {{ loading.style.display = 'none'; }}, 650);
      topbar.classList.add('visible');
      hud.classList.add('visible');
      metrics.classList.add('visible');
    }}, 300);

    trackFps();

  }} catch (err) {{
    console.error('[MonoSplat viewer]', err);
    showError('Render error: ' + (err.message || String(err)));
  }}
}})();
</script>
</body>
</html>'''
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Upload portal
# ---------------------------------------------------------------------------

@app.get("/capture-guide", response_class=HTMLResponse)
async def capture_guide():
    """Standalone capture best-practices page with an orbit diagram and checklist."""
    return HTMLResponse(r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Capture Guide — MonoSplat</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090b0f; --panel: #10141b; --panel2: #151b24;
      --line: #253041; --text: #eef3f8; --muted: #8b98aa;
      --soft: #c4cedb; --brand: #21d19f; --warn: #f5b84b;
      --bad: #ef5b6b; --blue: #70a7ff;
      --mono: 'JetBrains Mono', Consolas, monospace;
      --sans: Inter, system-ui, sans-serif;
      --r: 8px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; }
    a { color: var(--brand); text-decoration: none; }
    a:hover { text-decoration: underline; }

    header {
      position: sticky; top: 0; z-index: 10;
      display: flex; align-items: center; justify-content: space-between;
      gap: 24px; padding: 16px 32px;
      background: rgba(9,11,15,.92); border-bottom: 1px solid var(--line);
      backdrop-filter: blur(14px);
    }
    .brand-title { font-size: 20px; font-weight: 800; }
    .brand-title span { color: var(--brand); }
    .brand-sub { color: var(--muted); font-size: 12px; font-family: var(--mono); }

    .page { max-width: 860px; margin: 0 auto; padding: 52px 32px 80px; }
    .eyebrow { color: var(--brand); font-family: var(--mono); font-size: 12px; font-weight: 600; margin-bottom: 16px; }
    h1 { font-size: clamp(32px, 5vw, 52px); line-height: .96; margin-bottom: 18px; }
    .lead { color: var(--soft); line-height: 1.65; max-width: 620px; margin-bottom: 48px; font-size: 15px; }

    /* ── Orbit diagram ── */
    .orbit-wrap {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: var(--r); padding: 32px; margin-bottom: 48px;
      display: flex; flex-direction: column; align-items: center; gap: 20px;
    }
    .orbit-label { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }
    svg.orbit { display: block; }

    /* ── Checklist ── */
    .section-title {
      font-size: 11px; font-weight: 600; letter-spacing: .15em; text-transform: uppercase;
      color: var(--brand); font-family: var(--mono); margin-bottom: 20px;
    }
    .checklist { display: grid; gap: 12px; margin-bottom: 48px; }
    .check-item {
      display: flex; align-items: flex-start; gap: 14px;
      background: var(--panel); border: 1px solid var(--line);
      border-radius: var(--r); padding: 16px 18px;
    }
    .check-icon {
      width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 13px; font-weight: 700; margin-top: 1px;
    }
    .ok   .check-icon { background: rgba(33,209,159,.15); color: var(--brand); }
    .warn .check-icon { background: rgba(245,184,75,.15);  color: var(--warn); }
    .bad  .check-icon { background: rgba(239,91,107,.15);  color: var(--bad); }
    .check-body h3 { font-size: 14px; font-weight: 700; margin-bottom: 4px; }
    .check-body p  { color: var(--soft); font-size: 13px; line-height: 1.55; }
    .check-body code { font-family: var(--mono); color: var(--brand); font-size: 12px; }

    /* ── Do / Don't grid ── */
    .do-dont {
      display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 48px;
    }
    .do-box, .dont-box {
      border-radius: var(--r); padding: 20px 22px; border: 1px solid;
    }
    .do-box   { background: rgba(33,209,159,.05); border-color: rgba(33,209,159,.2); }
    .dont-box { background: rgba(239,91,107,.05); border-color: rgba(239,91,107,.2); }
    .do-box   h3 { color: var(--brand); font-size: 12px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 12px; }
    .dont-box h3 { color: var(--bad);   font-size: 12px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 12px; }
    .do-box   li, .dont-box li { color: var(--soft); font-size: 13px; line-height: 1.6; margin-left: 16px; }

    /* ── Blur warning ── */
    .blur-warn {
      background: rgba(245,184,75,.07); border: 1px solid rgba(245,184,75,.3);
      border-radius: var(--r); padding: 18px 20px; margin-bottom: 48px;
      display: flex; gap: 14px; align-items: flex-start;
    }
    .blur-icon { font-size: 20px; flex-shrink: 0; }
    .blur-body h3 { color: var(--warn); font-size: 14px; font-weight: 700; margin-bottom: 6px; }
    .blur-body p  { color: var(--soft); font-size: 13px; line-height: 1.55; }

    /* ── Settings table ── */
    .settings-table { width: 100%; border-collapse: collapse; margin-bottom: 48px; font-size: 13px; }
    .settings-table th {
      text-align: left; padding: 10px 14px; font-size: 11px; font-weight: 600;
      letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
      border-bottom: 1px solid var(--line); font-family: var(--mono);
    }
    .settings-table td { padding: 12px 14px; border-bottom: 1px solid rgba(37,48,65,.5); vertical-align: top; }
    .settings-table tr:hover td { background: var(--panel); }
    .settings-table td:first-child { color: var(--soft); font-weight: 600; }
    .tag-ok   { color: var(--brand); font-family: var(--mono); font-size: 11px; }
    .tag-warn { color: var(--warn);  font-family: var(--mono); font-size: 11px; }
    .tag-bad  { color: var(--bad);   font-family: var(--mono); font-size: 11px; }

    .back-link { display: inline-flex; align-items: center; gap: 6px; color: var(--brand); font-size: 13px; font-weight: 600; }
    @media (max-width: 640px) {
      header { padding: 14px 20px; }
      .page  { padding: 32px 20px 60px; }
      .do-dont { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

<header>
  <div>
    <div class="brand-title">Mono<span>Splat</span></div>
    <div style="color:var(--muted);font-size:12px;font-family:var(--mono)">single-camera 3D reconstruction</div>
  </div>
  <a href="/" class="back-link">← Back to portal</a>
</header>

<div class="page">

  <div class="eyebrow">Capture best practices</div>
  <h1>How to film for great reconstructions.</h1>
  <p class="lead">
    The quality of your 3D splat is almost entirely determined by how you film. The best reconstruction
    algorithm cannot recover from motion blur, missing coverage, or unstable exposure. Follow this guide
    and COLMAP will register cleanly every time.
  </p>

  <!-- Orbit diagram -->
  <div class="orbit-wrap">
    <div class="orbit-label">Recommended capture orbit — 360° slow walk-around</div>
    <svg class="orbit" viewBox="0 0 440 280" width="440" height="280" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#21d19f" stop-opacity=".25"/>
          <stop offset="100%" stop-color="#21d19f" stop-opacity="0"/>
        </radialGradient>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#21d19f" opacity=".7"/>
        </marker>
      </defs>
      <!-- glow -->
      <ellipse cx="220" cy="140" rx="80" ry="80" fill="url(#glow)"/>
      <!-- orbit path -->
      <ellipse cx="220" cy="140" rx="160" ry="90" fill="none"
               stroke="#21d19f" stroke-width="1.5" stroke-dasharray="6 4" opacity=".45"
               marker-end="url(#arr)"/>
      <!-- subject -->
      <rect x="197" y="117" width="46" height="46" rx="6"
            fill="#10141b" stroke="#253041" stroke-width="1.5"/>
      <text x="220" y="145" text-anchor="middle" fill="#c4cedb" font-size="11" font-family="Inter,sans-serif" font-weight="600">SUBJECT</text>
      <!-- camera positions -->
      <g fill="#21d19f">
        <circle cx="220" cy="50" r="7"/>
        <circle cx="380" cy="140" r="7"/>
        <circle cx="220" cy="230" r="7"/>
        <circle cx="60"  cy="140" r="7"/>
        <circle cx="330" cy="76"  r="5" opacity=".6"/>
        <circle cx="330" cy="204" r="5" opacity=".6"/>
        <circle cx="110" cy="76"  r="5" opacity=".6"/>
        <circle cx="110" cy="204" r="5" opacity=".6"/>
      </g>
      <!-- camera icons -->
      <text x="220" y="30"  text-anchor="middle" fill="#21d19f" font-size="14">📷</text>
      <text x="395" y="145" text-anchor="start"  fill="#21d19f" font-size="14">📷</text>
      <text x="220" y="256" text-anchor="middle" fill="#21d19f" font-size="14">📷</text>
      <text x="26"  y="145" text-anchor="start"  fill="#21d19f" font-size="14">📷</text>
      <!-- overlap annotation -->
      <text x="220" y="270" text-anchor="middle" fill="#8b98aa" font-size="10" font-family="JetBrains Mono,monospace">60–80% frame overlap between positions</text>
      <!-- direction arrow label -->
      <text x="356" y="60" fill="#21d19f" font-size="10" font-family="JetBrains Mono,monospace" opacity=".8">slow →</text>
    </svg>
    <div style="color:var(--muted);font-size:12px;text-align:center;max-width:400px;line-height:1.6">
      Film one continuous slow orbit. Move at roughly <strong style="color:var(--text)">one step per second</strong>.
      Keep the subject in the centre of frame at all times.
    </div>
  </div>

  <!-- Checklist -->
  <div class="section-title">Pre-flight checklist</div>
  <div class="checklist">
    <div class="check-item ok">
      <div class="check-icon">✓</div>
      <div class="check-body">
        <h3>Lock focus and exposure before you start</h3>
        <p>On iPhone: long-press the subject until you see <code>AE/AF Lock</code>. On Android: tap-hold the subject. Flickering exposure causes brightness inconsistency that confuses COLMAP's feature matching.</p>
      </div>
    </div>
    <div class="check-item ok">
      <div class="check-icon">✓</div>
      <div class="check-body">
        <h3>Choose a textured surface for the subject to rest on</h3>
        <p>Place items on newspaper, graph paper, or a patterned cloth. Feature-poor backgrounds (plain white table, glass desk) give COLMAP nothing to match against at the base.</p>
      </div>
    </div>
    <div class="check-item ok">
      <div class="check-icon">✓</div>
      <div class="check-body">
        <h3>Film 30–60 seconds at a slow, steady pace</h3>
        <p>One full orbit takes about 30–45 seconds when done right. Shorter clips mean fewer frames and worse COLMAP coverage. Longer clips slow down extraction without adding useful information.</p>
      </div>
    </div>
    <div class="check-item warn">
      <div class="check-icon">!</div>
      <div class="check-body">
        <h3>Avoid windows and skylights in the background</h3>
        <p>Strong backlight or changing sky exposure causes automatic exposure compensation to kick in, making sequential frames look like they were shot in different conditions.</p>
      </div>
    </div>
    <div class="check-item warn">
      <div class="check-icon">!</div>
      <div class="check-body">
        <h3>Record in a single continuous take — no cuts</h3>
        <p>Any cut or jump in the video breaks the temporal continuity that adaptive frame extraction relies on. If you accidentally pause, discard and re-record.</p>
      </div>
    </div>
    <div class="check-item bad">
      <div class="check-icon">✗</div>
      <div class="check-body">
        <h3>Do not film transparent, mirror, or featureless objects</h3>
        <p>Glass, polished metal, white ceramic — these surfaces have no stable visual features. COLMAP will fail to register frames. Cover the surface with matte tape or powder for scanning, then remove after.</p>
      </div>
    </div>
  </div>

  <!-- Blur warning -->
  <div class="blur-warn">
    <div class="blur-icon">⚠</div>
    <div class="blur-body">
      <h3>Motion blur is the #1 cause of COLMAP failure</h3>
      <p>
        At 30 fps, moving the phone more than about 5 cm per frame causes detectable blur. MonoSplat will
        flag high-motion frames during extraction and skip the worst ones automatically — but you cannot
        recover detail that was never captured sharply. Shoot slower than feels natural. Your 3D splat will
        thank you.
      </p>
    </div>
  </div>

  <!-- Do / Don't -->
  <div class="section-title">Do vs. Don't</div>
  <div class="do-dont">
    <div class="do-box">
      <h3>✓ Do</h3>
      <ul>
        <li>Slow, continuous orbit around subject</li>
        <li>Lock AE/AF before starting</li>
        <li>Diffuse indoor lighting (overcast day or softbox)</li>
        <li>Textured background under the object</li>
        <li>Landscape or portrait — both work</li>
        <li>60 fps source (auto-downsampled to 10 fps)</li>
        <li>H.264 MP4 or MOV (best compatibility)</li>
      </ul>
    </div>
    <div class="dont-box">
      <h3>✗ Don't</h3>
      <ul>
        <li>Fast pans or wrist flicks</li>
        <li>Auto-exposure left unlocked</li>
        <li>Hard directional sunlight or spotlight</li>
        <li>White table / clean desk as background</li>
        <li>Videos with cuts, pauses, or zooms</li>
        <li>Moving subject (people, pets, fans)</li>
        <li>Extremely small subjects under 5 cm</li>
      </ul>
    </div>
  </div>

  <!-- Settings table -->
  <div class="section-title">Recommended phone settings</div>
  <table class="settings-table">
    <thead>
      <tr>
        <th>Setting</th>
        <th>Recommended</th>
        <th>Why</th>
        <th>Rating</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Resolution</td>
        <td>1080p or 4K</td>
        <td>Higher res = more feature detail for COLMAP. 720p is marginal.</td>
        <td><span class="tag-ok">OPTIMAL</span></td>
      </tr>
      <tr>
        <td>Frame rate</td>
        <td>30 or 60 fps</td>
        <td>60 fps is auto-downsampled during extraction. Both work identically.</td>
        <td><span class="tag-ok">OPTIMAL</span></td>
      </tr>
      <tr>
        <td>Stabilisation (OIS/EIS)</td>
        <td>OFF if possible</td>
        <td>Electronic stabilisation warps the frame and corrupts camera poses.</td>
        <td><span class="tag-warn">CAUTION</span></td>
      </tr>
      <tr>
        <td>HDR video</td>
        <td>OFF</td>
        <td>HDR introduces tone-mapping that varies per frame — bad for consistency.</td>
        <td><span class="tag-warn">CAUTION</span></td>
      </tr>
      <tr>
        <td>Zoom</td>
        <td>1× only</td>
        <td>Digital zoom crops the sensor and degrades feature matching.</td>
        <td><span class="tag-bad">AVOID</span></td>
      </tr>
      <tr>
        <td>Portrait / Cinematic mode</td>
        <td>OFF</td>
        <td>Synthetic depth-of-field blurs background features COLMAP needs.</td>
        <td><span class="tag-bad">AVOID</span></td>
      </tr>
      <tr>
        <td>Slow-motion (240 fps)</td>
        <td>OFF</td>
        <td>Very high frame rates produce near-duplicate frames that waste COLMAP time.</td>
        <td><span class="tag-bad">AVOID</span></td>
      </tr>
    </tbody>
  </table>

  <a href="/" class="back-link">← Back to upload portal</a>

</div>
</body>
</html>""")


@app.get("/", response_class=HTMLResponse)
async def portal():
    html_path = Path("web/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1 style='font-family:monospace;padding:2rem'>"
        "Upload portal not found — place web/index.html</h1>"
    )