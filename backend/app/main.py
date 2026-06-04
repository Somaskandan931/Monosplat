"""
backend/app/main.py
--------------------
MonoSplat FastAPI application.

Flow:
  POST /upload                  → upload video, starts preprocessing pipeline
  GET  /status/{job_id}         → poll job status
  POST /upload-results/{job_id} → upload Colab training ZIP
  GET  /results/{job_id}        → get ply/splat paths for viewer

Run:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.database.session import Base, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("monosplat.api")

# ---------------------------------------------------------------------------
# Ensure required data directories exist BEFORE StaticFiles mount.
# StaticFiles raises RuntimeError at import time if the directory is missing,
# so mkdir must happen at module level — not inside the lifespan handler.
# ---------------------------------------------------------------------------
for _dir in ("data/uploads", "data/outputs", "data/results"):
    Path(_dir).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("MonoSplat API starting — initialising database")
    Base.metadata.create_all(bind=engine)
    log.info("Ready")
    yield
    log.info("MonoSplat API shutting down")


app = FastAPI(
    title="MonoSplat API",
    description=(
        "REST API for MonoSplat — Monocular 3D Gaussian Splatting pipeline.\n\n"
        "**Flow**: Upload video → pipeline preprocesses → package for Colab "
        "→ user trains on Colab → upload results → view in browser."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Mounted at /static/results to avoid shadowing GET /results/{job_id}.
# The directory is guaranteed to exist by the mkdir block above.
app.mount("/static/results", StaticFiles(directory="data/results", html=False), name="results")


@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok", "service": "monosplat-api"})