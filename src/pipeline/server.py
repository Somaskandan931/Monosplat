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

Endpoints
---------
    GET  /                              Upload portal HTML
    POST /upload                        Upload video for a named scene
    GET  /api/jobs                      List all jobs + statuses
    GET  /api/jobs/{job_id}             Single job status
    GET  /api/jobs/{job_id}/stream      SSE live status stream (stage + progress)
    GET  /api/jobs/{job_id}/metrics     PSNR / SSIM / timing for a completed job
    GET  /api/models                    List READY models
    GET  /api/models/latest             Most recently completed model
    GET  /api/health                    System health (queue mode, Redis, workers)
    GET  /splat/{job_id}                Serve .splat binary for Three.js viewer
    GET  /ply/{job_id}                  Serve .ply for download
    GET  /thumbnails/{job_id}           Serve thumbnail PNG
    GET  /viewer/{job_id}               Inline Three.js Gaussian splat viewer
"""

import asyncio
import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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
    print("[server] MonoSplat server started ✓")
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

    return JSONResponse({
        "status":       "ok",
        "uptime_s":     round(time.time() - _START_TIME, 1),
        "queue_mode":   get_mode(),
        "redis_ok":     redis_ok,
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
# File serving
# ---------------------------------------------------------------------------

@app.get("/thumbnails/{job_id}")
async def get_thumbnail(job_id: str):
    job = manager.get_registry().get_job(job_id)
    if not job or not job.thumbnail or not Path(job.thumbnail).exists():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(job.thumbnail, media_type="image/png")


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

    if job.splat_path is None:
        return HTMLResponse(
            f"<h2 style='font-family:monospace;padding:2rem'>"
            f"Model not ready yet — status: {job.status}</h2>"
        )

    splat_url = f"/splat/{job_id}"
    item_name = job.item_name

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{item_name} — MonoSplat Viewer</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body   {{ background: #0a0c0f; color: #fff; font-family: monospace; overflow: hidden; }}
    canvas {{ display: block; width: 100vw; height: 100vh; }}
    #hud   {{
      position: fixed; top: 16px; left: 16px;
      background: rgba(10,12,15,0.85); border: 1px solid #1e2530;
      border-radius: 6px; padding: 14px 18px; font-size: 12px; line-height: 1.9;
      pointer-events: none;
    }}
    #hud h2     {{ font-size: 15px; color: #00e5a0; margin-bottom: 6px; }}
    #hud span   {{ color: #4a5568; }}
    #metrics    {{
      position: fixed; bottom: 16px; left: 16px;
      background: rgba(10,12,15,0.85); border: 1px solid #1e2530;
      border-radius: 6px; padding: 10px 16px; font-size: 11px; line-height: 1.8;
      pointer-events: none;
    }}
    #metrics .label {{ color: #4a5568; }}
    #metrics .val   {{ color: #00e5a0; margin-left: 8px; }}
    #loading    {{
      position: fixed; inset: 0; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 16px;
      background: #0a0c0f; z-index: 100;
    }}
    #loading p  {{ color: #00e5a0; font-size: 14px; letter-spacing: .1em; }}
    .bar-track  {{ width: 240px; height: 3px; background: #1e2530; border-radius: 2px; overflow: hidden; }}
    .bar-fill   {{ height: 100%; width: 0; background: #00e5a0; transition: width .3s; }}
    #back       {{
      position: fixed; top: 16px; right: 16px;
      background: rgba(10,12,15,0.85); border: 1px solid #1e2530;
      border-radius: 4px; padding: 8px 14px; font-size: 11px;
      color: #4a5568; text-decoration: none; letter-spacing: .08em; text-transform: uppercase;
    }}
    #back:hover {{ color: #00e5a0; border-color: #00e5a0; }}
  </style>
</head>
<body>

<div id="loading">
  <p>LOADING SPLAT MODEL</p>
  <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
  <p id="msg" style="color:#4a5568;font-size:11px">Fetching {item_name}…</p>
</div>

<div id="hud" style="display:none">
  <h2>{item_name}</h2>
  <span>Left drag</span>  Rotate<br>
  <span>Right drag</span> Pan<br>
  <span>Scroll</span>     Zoom<br>
  <span>R</span>          Reset camera
</div>

<div id="metrics" style="display:none">
  <span class="label">PSNR</span><span class="val" id="m-psnr">—</span><br>
  <span class="label">SSIM</span><span class="val" id="m-ssim">—</span><br>
  <span class="label">Gaussians</span><span class="val" id="m-gauss">—</span>
</div>

<a id="back" href="/">← Back</a>

<script type="module">
import * as GaussianSplats3D from
  'https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@latest/build/gaussian-splats-3d.module.js';

const loading = document.getElementById('loading');
const bar     = document.getElementById('bar');
const msg     = document.getElementById('msg');
const hud     = document.getElementById('hud');
const metrics = document.getElementById('metrics');

// Load quality metrics from API
fetch('/api/jobs/{job_id}/metrics')
  .then(r => r.json())
  .then(d => {{
    if (d.psnr) document.getElementById('m-psnr').textContent = d.psnr + ' dB';
    if (d.ssim) document.getElementById('m-ssim').textContent = d.ssim;
    if (d.num_gaussians) document.getElementById('m-gauss').textContent =
      (d.num_gaussians / 1e6).toFixed(2) + 'M';
  }}).catch(() => {{}});

const viewer = new GaussianSplats3D.Viewer({{
  cameraUp:              [0, -1, 0],
  initialCameraPosition: [0,  0,  5],
  initialCameraLookAt:   [0,  0,  0],
  selfDrivenMode:        true,
}});

msg.textContent = 'Downloading splat data…';
bar.style.width = '20%';

viewer.addSplatScene('{splat_url}', {{
  splatAlphaRemovalThreshold: 5,
  onProgress: pct => {{
    bar.style.width = (20 + pct * 0.8) + '%';
    msg.textContent = 'Loading… ' + Math.round(pct) + '%';
  }},
}})
.then(() => {{
  bar.style.width = '100%';
  setTimeout(() => {{
    loading.style.display = 'none';
    hud.style.display = 'block';
    metrics.style.display = 'block';
  }}, 300);
  viewer.start();
}})
.catch(err => {{
  msg.textContent  = 'Error: ' + err.message;
  msg.style.color  = '#ff4757';
}});

document.addEventListener('keydown', e => {{
  if (e.key === 'r' || e.key === 'R') {{
    viewer.getCamera().position.set(0, 0, 5);
    viewer.getCamera().lookAt(0, 0, 0);
  }}
}});
</script>
</body>
</html>'''
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Upload portal
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def portal():
    html_path = Path("web/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1 style='font-family:monospace;padding:2rem'>"
        "Upload portal not found — place web/index.html</h1>"
    )
