"""
verify_pipeline.py
Check that all three MonoSplat pipeline stages can run on this machine.

Stage 1 — FFmpeg frame extraction
Stage 2 — COLMAP SfM pose estimation
Stage 3 — PyTorch Gaussian training (CUDA or Colab handoff)

Usage:
    python scripts/verify_pipeline.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.console import configure_console_encoding
from src.utils.config_loader import load_config
from src.utils.env_detect import get_env_info, has_torch_gpu

configure_console_encoding()


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    print("MonoSplat pipeline verification\n")
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    env = get_env_info()
    all_ok = True

    # ── Stage 1: FFmpeg ───────────────────────────────────────────────────
    print("Stage 1 — Frames (FFmpeg)")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    all_ok &= check("FFmpeg on PATH", bool(ffmpeg), ffmpeg or "not found")
    all_ok &= check("ffprobe on PATH", bool(ffprobe), ffprobe or "not found")

    try:
        import cv2
        all_ok &= check("opencv-python", True, cv2.__version__)
    except ImportError:
        all_ok &= check("opencv-python", False, "pip install opencv-python")

    try:
        from src.preprocessing.extract_frames import _check_ffmpeg
        _check_ffmpeg()
        all_ok &= check("extract_frames module", True)
    except Exception as e:
        all_ok &= check("extract_frames module", False, str(e))

    # ── Stage 2: COLMAP ───────────────────────────────────────────────────
    print("\nStage 2 — Poses (COLMAP SfM)")
    colmap_bin = getattr(cfg.colmap, "binary_path", "colmap")
    colmap_path = shutil.which(colmap_bin)
    all_ok &= check("COLMAP on PATH", bool(colmap_path), colmap_path or colmap_bin)

    if colmap_path:
        try:
            r = subprocess.run(
                [colmap_path, "-h"],
                capture_output=True, text=True, timeout=15,
            )
            ver = (r.stdout or r.stderr).splitlines()[0][:80]
            all_ok &= check("COLMAP runs", r.returncode == 0, ver)
        except Exception as e:
            all_ok &= check("COLMAP runs", False, str(e))

    # CPU COLMAP on Windows is expected and correct (GPU SIFT often crashes)
    ext_detail = "GPU enabled" if env["colmap_extraction_gpu"] else "CPU mode (recommended on Windows)"
    match_detail = "GPU enabled" if env["colmap_matching_gpu"] else "CPU mode (recommended on Windows)"
    check("COLMAP extraction", True, ext_detail)
    check("COLMAP matching", True, match_detail)

    try:
        from src.preprocessing.colmap_runner import run_colmap
        all_ok &= check("colmap_runner module", True)
    except Exception as e:
        all_ok &= check("colmap_runner module", False, str(e))

    # ── Stage 3: Training ─────────────────────────────────────────────────
    print("\nStage 3 — Train (PyTorch Gaussian Splat)")
    try:
        import torch
        all_ok &= check("PyTorch", True, torch.__version__)
        cuda = torch.cuda.is_available()
        all_ok &= check(
            "CUDA GPU (local training)",
            cuda,
            torch.cuda.get_device_name(0) if cuda else "use Colab — ready_for_colab after COLMAP",
        )
    except ImportError:
        all_ok &= check("PyTorch", False, "pip install torch")

    try:
        from src.reconstruction.trainer import GaussianTrainer
        from src.reconstruction.gaussian_model import GaussianModel
        all_ok &= check("trainer module", True)
    except Exception as e:
        all_ok &= check("trainer module", False, str(e))

    all_ok &= check("train.py script", (ROOT / "scripts" / "train.py").exists())
    all_ok &= check("Colab notebook", (ROOT / "notebooks" / "monosplat_colab_gpu.ipynb").exists())
    all_ok &= check("zip_for_colab.py", (ROOT / "scripts" / "zip_for_colab.py").exists())
    all_ok &= check("mark_ready.py", (ROOT / "scripts" / "mark_ready.py").exists())

    # ── Server / portal ───────────────────────────────────────────────────
    print("\nWeb portal")
    try:
        from src.pipeline.server import app
        routes = {r.path for r in app.routes}
        needed = {"/", "/upload", "/api/health", "/viewer/{job_id}", "/splat/{job_id}"}
        missing = needed - routes
        all_ok &= check("FastAPI routes", not missing, ", ".join(sorted(missing)) if missing else "all present")
    except Exception as e:
        all_ok &= check("FastAPI server", False, str(e))

    all_ok &= check("Upload portal HTML", (ROOT / "web" / "index.html").exists())

    # ── Config ────────────────────────────────────────────────────────────
    print("\nConfig (config/config.yaml)")
    all_ok &= check("video_fps", hasattr(cfg.data, "video_fps"), str(getattr(cfg.data, "video_fps", "?")))
    all_ok &= check("max_frames", hasattr(cfg.data, "max_frames"), str(getattr(cfg.data, "max_frames", "?")))
    all_ok &= check("colmap quality", hasattr(cfg.colmap, "quality"), getattr(cfg.colmap, "quality", "?"))
    all_ok &= check("training iterations", hasattr(cfg.training, "iterations"), str(getattr(cfg.training, "iterations", "?")))

    print()
    if all_ok:
        print("All checks passed. Run: python -m uvicorn src.pipeline.server:app --reload --port 8000")
        return 0

    print("Some checks failed — fix the items marked FAIL before running the full pipeline.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
