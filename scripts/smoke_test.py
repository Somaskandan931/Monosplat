"""
Smoke test for the MonoSplat product server.

This checks imports, configuration loading, FastAPI routes, and the upload portal
without requiring FFmpeg, COLMAP, CUDA, or a real video.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.pipeline.server import app


def main() -> int:
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    route_paths = {route.path for route in app.routes}
    required = {"/", "/upload", "/api/jobs", "/api/health", "/viewer/{job_id}", "/splat/{job_id}"}
    missing = sorted(required - route_paths)
    if missing:
        print("Missing routes:", ", ".join(missing))
        return 1

    portal = ROOT / "web" / "index.html"
    if not portal.exists() or "MonoSplat" not in portal.read_text(encoding="utf-8"):
        print("Upload portal is missing or invalid.")
        return 1

    print("MonoSplat smoke test passed")
    print(f"Project: {cfg.project.name}")
    print(f"Routes checked: {len(required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
