"""
scripts/prepare_dataset.py
==========================
Canonical LOCAL preprocessing entrypoint for MonoSplat.

Full pipeline:
    Video/Images → Frame Extraction → COLMAP Reconstruction → Dataset ZIP

Usage
-----
    # From a video file:
    python scripts/prepare_dataset.py --video input.mp4

    # From an existing folder of images:
    python scripts/prepare_dataset.py --images path/to/frames/

    # With explicit job ID:
    python scripts/prepare_dataset.py --video input.mp4 --job_id myshot

    # Override quality:
    python scripts/prepare_dataset.py --video input.mp4 --quality high

    # Skip ZIP creation (for local GPU training):
    python scripts/prepare_dataset.py --video input.mp4 --no_zip

Output
------
    work/<job_id>/
        frames/              ← extracted / validated frames
        colmap/
            database.db
            sparse/0/        ← binary COLMAP model
            sparse_text/     ← cameras.txt, images.txt, points3D.txt
        metrics.json
        logs.txt

FIXES APPLIED:
  [FIX-1] All cfg access converted from cfg.section.key (attribute access on dict
          → AttributeError) to cfg["section"]["key"] (dict access).
          The old code assumed load_config() returned an object with attributes
          (cfg.colmap, cfg.data) but load_config() returns a plain dict.
          config_loader.py now returns a _ConfigProxy that supports both styles,
          but this script uses the explicit dict style to be unambiguous.
  [FIX-2] sys.path setup matches train.py (adds _REPO_ROOT/src, not _REPO_ROOT).
  [FIX-3] PipelineMetrics import is guarded — if src/utils/metrics.py is missing
          the script still runs (metrics are optional diagnostics).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root / sys.path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config

# FIX-3: guard optional metrics import
try:
    from utils.metrics import PipelineMetrics
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False
    class PipelineMetrics:  # stub
        def __init__(self, **kw): pass
        def set_frame_metrics(self, **kw): pass
        def set_reconstruction_metrics(self, **kw): pass
        def mark_failed(self, reason=""): pass
        def mark_success(self, **kw): pass
        def save(self, **kw): pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_id_from_input(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _count_images(directory: Path) -> int:
    exts = {".jpg", ".jpeg", ".png"}
    return sum(
        1 for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def _validate_prereqs() -> None:
    import shutil
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg  (install: sudo apt install ffmpeg  /  brew install ffmpeg)")
    if not shutil.which("colmap"):
        missing.append("colmap  (install: sudo apt install colmap  /  brew install colmap)")
    if missing:
        print("\n[prepare] ✗  Missing required tools:")
        for m in missing:
            print(f"             • {m}")
        sys.exit(1)


def _validate_colmap_output(sparse_text: Path) -> None:
    required = ["cameras.txt", "images.txt", "points3D.txt"]
    for fname in required:
        fp = sparse_text / fname
        if not fp.exists():
            raise FileNotFoundError(f"[validate] COLMAP output missing: {fp}")
        if fp.stat().st_size == 0:
            raise RuntimeError(f"[validate] COLMAP file is empty: {fp}")

    registered = 0
    with open(sparse_text / "images.txt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) == 10:
                    registered += 1

    points_count = 0
    with open(sparse_text / "points3D.txt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                points_count += 1

    print(f"[validate] ✓  COLMAP sparse model: {registered} registered images, {points_count:,} 3D points")

    if registered < 10:
        raise RuntimeError(
            f"[validate] ✗  Only {registered} images registered by COLMAP (minimum: 10)."
        )
    if points_count < 100:
        raise RuntimeError(
            f"[validate] ✗  Only {points_count} 3D points reconstructed (minimum: 100)."
        )


def _build_zip(job_id: str, job_path: Path, config_path: Path) -> str:
    zip_name = f"{job_id}_for_colab.zip"
    frames_dir  = job_path / "frames"
    sparse_text = job_path / "colmap" / "sparse_text"

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not sparse_text.exists():
        raise FileNotFoundError(f"COLMAP sparse_text not found: {sparse_text}")

    frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    print(f"\n[zip] Packaging {len(frame_files)} frames + COLMAP model → {zip_name}")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in frame_files:
            zf.write(f, f"work/{job_id}/frames/{f.name}")
        for f in sorted(sparse_text.rglob("*")):
            if f.is_file():
                zf.write(f, f"work/{job_id}/colmap/sparse_text/{f.name}")
        sparse0 = job_path / "colmap" / "sparse" / "0"
        if sparse0.exists():
            for f in sorted(sparse0.rglob("*")):
                if f.is_file():
                    zf.write(f, f"work/{job_id}/colmap/sparse/0/{f.name}")
        if config_path.exists():
            zf.write(config_path, "config/config.yaml")
        metrics_path = job_path / "metrics.json"
        if metrics_path.exists():
            zf.write(metrics_path, f"work/{job_id}/metrics.json")
        quality_report_path = job_path / "quality_report.json"
        if quality_report_path.exists():
            zf.write(quality_report_path, f"work/{job_id}/quality_report.json")
        prediction_report_path = job_path / "prediction_report.json"
        if prediction_report_path.exists():
            zf.write(prediction_report_path, f"work/{job_id}/prediction_report.json")
        frame_selection_report_path = job_path / "frame_selection_report.json"
        if frame_selection_report_path.exists():
            zf.write(frame_selection_report_path, f"work/{job_id}/frame_selection_report.json")

    size_mb = Path(zip_name).stat().st_size / 1e6
    print(f"[zip] ✓  Created {zip_name}  ({size_mb:.1f} MB)")
    return zip_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_prepare_dataset(
    video: str | None = None,
    images: str | None = None,
    job_id: str | None = None,
    work_dir: str = "work",
    config: str | None = None,
    quality: str | None = None,
    max_frames: int | None = None,
    fps: float | None = None,
    no_zip: bool = False,
    skip_colmap: bool = False,
    force: bool = False,
) -> dict:
    """
    Programmatic entry point for the preprocessing pipeline.

    Equivalent to running ``python scripts/prepare_dataset.py`` but callable
    from Python code (e.g. the FastAPI backend) without spawning a subprocess.

    Returns a dict with keys: job_id, frames_dir, sparse_text, zip_path (or None).
    Raises RuntimeError on unrecoverable failure instead of calling sys.exit().
    """
    import sys as _sys

    # Build a fake argv so the argparse-based main() parses correctly.
    _argv = []
    if video:
        _argv += ["--video", video]
    elif images:
        _argv += ["--images", images]
    else:
        raise ValueError("Either video or images must be provided")
    if job_id:      _argv += ["--job_id",    job_id]
    if work_dir:    _argv += ["--work_dir",  work_dir]
    if config:      _argv += ["--config",    config]
    if quality:     _argv += ["--quality",   quality]
    if max_frames:  _argv += ["--max_frames", str(max_frames)]
    if fps:         _argv += ["--fps",        str(fps)]
    if no_zip:      _argv.append("--no_zip")
    if skip_colmap: _argv.append("--skip_colmap")
    if force:       _argv.append("--force")

    _orig_argv = _sys.argv
    try:
        _sys.argv = ["prepare_dataset.py"] + _argv
        main()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"prepare_dataset pipeline failed (exit {e.code})")
    finally:
        _sys.argv = _orig_argv

    # Reconstruct result paths from work_dir + job_id (main() prints them).
    from pathlib import Path as _Path
    import hashlib as _hashlib

    _input = _Path(video or images).resolve()
    _jid = job_id or _hashlib.md5(
        f"{_input}:{_input.stat().st_size}:{_input.stat().st_mtime}".encode()
    ).hexdigest()[:12]

    _job_path    = _Path(work_dir) / _jid
    _sparse_text = _job_path / "colmap" / "sparse_text"
    _frames_dir  = _job_path / "frames"
    _zip_path    = _Path(f"{_jid}_for_colab.zip") if not no_zip else None

    return {
        "job_id":      _jid,
        "frames_dir":  str(_frames_dir),
        "sparse_text": str(_sparse_text),
        "zip_path":    str(_zip_path) if _zip_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MonoSplat — local preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video",  metavar="FILE",  help="Input video (.mp4 / .mov / .avi / .mkv)")
    source.add_argument("--images", metavar="DIR",   help="Directory of input images")

    parser.add_argument("--job_id",       default=None)
    parser.add_argument("--work_dir",     default="work")
    parser.add_argument("--config",       default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--quality",      default=None, choices=["low", "medium", "high"])
    parser.add_argument("--max_frames",   type=int, default=None)
    parser.add_argument("--fps",          type=float, default=None)
    parser.add_argument("--no_zip",       action="store_true")
    parser.add_argument("--skip_colmap",  action="store_true")
    parser.add_argument("--force",        action="store_true")
    args = parser.parse_args()

    t_start = time.time()

    # FIX-1: load_config returns a dict (_ConfigProxy); use dict-style access
    cfg = load_config(args.config)
    try:
        from core.hardware import HardwareDetector, HardwareProfileManager
        hardware_report = HardwareDetector().detect()
        cfg, active_profile = HardwareProfileManager(PROJECT_ROOT / "configs").apply_profile(cfg, hardware_report)
        print(f"[prepare] Hardware profile: {active_profile} ({hardware_report.get('gpu_type')}, {hardware_report.get('vram_gb')}GB)")
    except Exception as e:
        print(f"[prepare] WARNING: Hardware profile detection failed; using config defaults: {e}")
    config_path = Path(args.config)

    # FIX-1: cfg["colmap"]["quality"] not cfg.colmap.quality
    quality    = args.quality    or cfg["colmap"].get("quality",    "medium")
    max_frames = args.max_frames or cfg["data"].get("max_frames", 300)

    input_path = Path(args.video or args.images).resolve()
    if not input_path.exists():
        print(f"[prepare] ✗  Input not found: {input_path}")
        sys.exit(1)

    job_id    = args.job_id or _job_id_from_input(input_path)
    work_root = Path(args.work_dir)
    job_path  = work_root / job_id

    frames_dir  = job_path / "frames"
    colmap_dir  = job_path / "colmap"
    sparse_text = colmap_dir / "sparse_text"

    job_path.mkdir(parents=True, exist_ok=True)
    log_path = job_path / "logs.txt"
    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    metrics = PipelineMetrics(job_id=job_id)

    _validate_prereqs()

    print("=" * 64)
    print("  MonoSplat — Preprocessing Pipeline")
    print("=" * 64)
    print(f"  Job ID    : {job_id}")
    print(f"  Input     : {input_path}")
    print(f"  Work dir  : {job_path.resolve()}")
    print(f"  Quality   : {quality}")
    print(f"  Max frames: {max_frames}")
    print("=" * 64)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Frame extraction
    # ════════════════════════════════════════════════════════════════════════
    import shutil as _shutil
    for _stale in ["blurry", "duplicates"]:
        _p = frames_dir / _stale
        if _p.exists():
            _shutil.rmtree(str(_p))

    if frames_dir.exists() and _count_images(frames_dir) > 0 and not args.force:
        n_frames = _count_images(frames_dir)
        print(f"\n[Step 1] Frames directory exists with {n_frames} images — skipping extraction.")
        _frames_reextracted = False
    else:
        _frames_reextracted = True
        if frames_dir.exists():
            _shutil.rmtree(str(frames_dir))
        print("\n[Step 1] Extracting frames…")
        frames_dir.mkdir(parents=True, exist_ok=True)

        from preprocessing.extract_frames import (
            extract_from_video, copy_images, validate_images,
            validate_image_resolution,
        )

        if args.video:
            n_frames = extract_from_video(
                video_path=str(input_path),
                output_dir=str(frames_dir),
                fps=args.fps,
                max_frames=max_frames,
                blur_threshold=80.0,
                adaptive_sampling=False,
            )
        else:
            n_frames = copy_images(
                image_dir=str(input_path),
                output_dir=str(frames_dir),
                max_frames=max_frames,
            )

        if n_frames == 0:
            print(f"[prepare] ✗  No frames extracted from {input_path}")
            metrics.mark_failed("frame_extraction_zero_output")
            metrics.save(work_dir=str(work_root))
            sys.exit(1)

        try:
            validate_image_resolution(str(frames_dir), min_size=256)
        except RuntimeError as e:
            print(f"\n[prepare] ✗  {e}")
            metrics.mark_failed(f"resolution_validation_failed: {e}")
            metrics.save(work_dir=str(work_root))
            sys.exit(1)

        print(f"[Step 1] ✓  {n_frames} frames ready in {frames_dir}")

    try:
        from preprocessing.extract_frames import run_smart_frame_selection
        selection_report = run_smart_frame_selection(str(frames_dir), budget=max_frames)
        n_frames = selection_report.get("selected_frame_count", n_frames)
    except Exception as e:
        print(f"[Step 1] WARNING: Smart frame selection failed; continuing with existing frames: {e}")

    metrics.set_frame_metrics(frame_count=n_frames, filtered_frames=n_frames)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1.5 — Dataset analysis before COLMAP
    # ════════════════════════════════════════════════════════════════════════
    print("\n[Step 1.5] Analyzing dataset quality before COLMAP...")
    quality_report_path = job_path / "quality_report.json"
    try:
        from core.dataset_analysis import DatasetAnalysisPipeline
        from core.quality_prediction import ReconstructionSuccessPredictor

        quality_report = DatasetAnalysisPipeline().analyze(
            frames_dir,
            output_path=quality_report_path,
        )
        prediction_report_path = job_path / "prediction_report.json"
        prediction_report = ReconstructionSuccessPredictor().predict(
            quality_report,
            output_path=prediction_report_path,
        )
        print(
            "[Step 1.5] Quality report: "
            f"success_probability={quality_report['success_probability']:.2f}, "
            f"recommended_frames={quality_report['recommended_frames']}, "
            f"blur={quality_report['blur_score']:.2f}, "
            f"coverage={quality_report['coverage_score']:.2f}, "
            f"texture={quality_report['texture_score']:.2f}"
        )
        print(f"[Step 1.5] Saved quality_report.json -> {quality_report_path}")
        print(
            "[Step 1.5] Prediction: "
            f"risk={prediction_report['risk_level']} "
            f"success_probability={prediction_report['success_probability']:.2f}"
        )
        print(f"[Step 1.5] Saved prediction_report.json -> {prediction_report_path}")
        if prediction_report["risk_level"] == "high":
            print("[Step 1.5] HIGH RISK dataset detected before COLMAP:")
            for reason in prediction_report["explanation"]["risk_factors"]:
                print(f"  - {reason}")
            print(f"  Recommended action: {prediction_report['recommended_action']}")
    except Exception as e:
        print(f"[Step 1.5] WARNING: Dataset analysis failed; continuing to COLMAP: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — COLMAP reconstruction
    # ════════════════════════════════════════════════════════════════════════
    if _frames_reextracted and colmap_dir.exists() and not args.skip_colmap:
        _shutil.rmtree(str(colmap_dir))

    if args.skip_colmap and sparse_text.exists():
        try:
            _validate_colmap_output(sparse_text)
            print("\n[Step 2] COLMAP output exists — skipping reconstruction.")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n[prepare] ✗  Existing COLMAP output invalid: {e}")
            sys.exit(1)
    else:
        print("\n[Step 2] Running COLMAP reconstruction…")

        from preprocessing.colmap_runner import run_colmap

        # FIX-1: dict access, not attribute access
        try:
            stats = run_colmap(
                image_dir=str(frames_dir),
                output_dir=str(colmap_dir),
                colmap_binary=cfg["colmap"].get("binary_path", "colmap"),
                camera_model=cfg["colmap"].get("camera_model", "OPENCV"),
                single_camera=cfg["colmap"].get("single_camera", True),
                quality=quality,
                use_gpu=True,
                force_gpu=False,
            )
        except RuntimeError as e:
            print(f"\n[prepare] ✗  COLMAP failed:\n  {e}")
            metrics.mark_failed(f"colmap_failed: {e}")
            metrics.save(work_dir=str(work_root))
            sys.exit(1)

        try:
            _validate_colmap_output(sparse_text)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n[prepare] ✗  COLMAP output validation failed:\n  {e}")
            metrics.mark_failed(f"colmap_output_invalid: {e}")
            metrics.save(work_dir=str(work_root))
            sys.exit(1)

        registered = stats.get("registered", 0)
        n_points   = stats.get("n_points",   0)
        tier_used  = stats.get("tier_used",  "unknown")

        metrics.set_reconstruction_metrics(
            registered_images=registered,
            total_images=n_frames,
            sparse_points=n_points,
        )
        print(f"[Step 2] ✓  COLMAP done: {registered}/{n_frames} images registered, "
              f"{n_points:,} 3D points (tier: {tier_used})")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — Dataset integrity verification
    # ════════════════════════════════════════════════════════════════════════
    print("\n[Step 3] Verifying dataset integrity…")
    missing_images = []
    with open(sparse_text / "images.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 10:
                img_name = parts[9]
                if not (frames_dir / img_name).exists():
                    alt = frames_dir / Path(img_name).name
                    if not alt.exists():
                        missing_images.append(img_name)

    if missing_images:
        print(f"[prepare] ⚠   {len(missing_images)} images in COLMAP not found in frames/")
        for name in missing_images[:5]:
            print(f"              • {name}")
    else:
        print("[Step 3] ✓  All COLMAP-referenced images present in frames/")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 — Package for Colab
    # ════════════════════════════════════════════════════════════════════════
    zip_name = None
    if not args.no_zip:
        print("\n[Step 4] Creating Colab upload package…")
        try:
            zip_name = _build_zip(job_id, job_path, config_path)
        except Exception as e:
            print(f"[prepare] ✗  ZIP creation failed: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    metrics.mark_success(wall_seconds=elapsed)
    metrics.save(work_dir=str(work_root))

    print("\n" + "=" * 64)
    print(f"  ✅  Preprocessing complete in {elapsed:.0f}s")
    print("=" * 64)
    print(f"  Job ID        : {job_id}")
    print(f"  Frames        : {frames_dir.resolve()}  ({_count_images(frames_dir)} frames)")
    print(f"  COLMAP output : {sparse_text.resolve()}")
    if zip_name:
        print(f"  Colab ZIP     : {Path(zip_name).resolve()}")
    print("=" * 64)
    print("\nNext step — Local GPU training:")
    print(f"    python scripts/train.py \\")
    print(f"        --sparse {sparse_text} \\")
    print(f"        --frames {frames_dir} \\")
    print(f"        --output {job_path / 'models' / 'gaussian'}")
    print()


if __name__ == "__main__":
    main()
