"""
Mark a job as ready after Colab GPU training and placing output files locally.

Usage:
    python scripts/mark_ready.py <job_id>

Expects:
    work/<job_id>/models/gaussian/<job_id>.splat
    work/<job_id>/models/gaussian/<job_id>.ply   (optional)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.console import configure_console_encoding
from src.utils.io_utils import load_ply, save_splat, splat_bounds

configure_console_encoding()


def mark_ready(job_id: str, reencode_splat: bool = True) -> None:
    work = ROOT / "work" / job_id
    gaussian_dir = work / "models" / "gaussian"
    splat_path = gaussian_dir / f"{job_id}.splat"
    ply_path = gaussian_dir / f"{job_id}.ply"
    registry_path = ROOT / "models" / "registry.json"

    if not splat_path.exists() and ply_path.exists() and reencode_splat:
        print(f"[mark_ready] Converting {ply_path.name} -> {splat_path.name}")
        save_splat(str(splat_path), load_ply(str(ply_path)))

    if not splat_path.exists():
        raise FileNotFoundError(
            f"No .splat at {splat_path}\n"
            f"Train in Colab, download {job_id}.splat, and place it in:\n"
            f"  {gaussian_dir}"
        )

    if reencode_splat and ply_path.exists():
        print(f"[mark_ready] Re-encoding .splat for browser viewer compatibility")
        save_splat(str(splat_path), load_ply(str(ply_path)))

    bounds = splat_bounds(str(splat_path))

    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    if job_id not in registry:
        raise KeyError(f"Job {job_id} not in registry")

    rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/")
    registry[job_id].update({
        "status": "ready",
        "progress": 100,
        "message": f"Ready — {bounds['num_splats']:,} splats (view in browser)",
        "splat_path": rel(splat_path),
        "ply_path": rel(ply_path) if ply_path.exists() else None,
        "num_gaussians": bounds["num_splats"],
        "error": None,
    })

    import os
    import shutil
    tmp = registry_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    try:
        tmp.replace(registry_path)
    except PermissionError:
        if registry_path.exists():
            os.remove(registry_path)
        shutil.move(str(tmp), str(registry_path))

    print(f"[mark_ready] Job {job_id} is ready")
    print(f"  Splats: {bounds['num_splats']:,}")
    print(f"  Viewer: http://localhost:8000/viewer/{job_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark a Colab-trained job as ready")
    parser.add_argument("job_id", help="Job ID from models/registry.json")
    parser.add_argument("--no-reencode", action="store_true", help="Skip .splat re-encoding")
    args = parser.parse_args()
    mark_ready(args.job_id, reencode_splat=not args.no_reencode)
