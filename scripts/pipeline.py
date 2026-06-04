"""
scripts/pipeline.py
-------------------
MonoSplat — preprocessing pipeline orchestrator.

Callable from the backend (or any Python code):

    from scripts.pipeline import run_pipeline
    result = run_pipeline("video.mp4")
    print(result["zip"])       # path to Colab upload package
    print(result["dataset"])   # path to sparse_text dir

Training lives in colab/train.py and runs on Google Colab.
core/ intelligence is an opt-in gate (see run_pipeline's `use_quality_gate`).
"""

from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path

log = logging.getLogger("monosplat.pipeline")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str,
    output_root: str = "outputs",
    fps: float = 2.0,
    max_frames: int = 150,
    quality: str = "medium",
    use_gpu: bool = True,
    use_quality_gate: bool = False,   # set True to enable core/ gating
) -> dict:
    """
    Run the full local preprocessing pipeline.

    Steps
    -----
    1. Frame extraction          (src/preprocessing/extract_frames.py)
    2. Smart frame selection     (core/frame_selection)
    3. [Optional] Quality gate   (core/dataset_analysis + quality_prediction)
    4. COLMAP sparse recon       (src/preprocessing/colmap_runner.py)
    5. Colab package ZIP         (inline — no extra dependency)

    Returns
    -------
    dict with keys:
        "frames"   — absolute path to extracted frames dir
        "dataset"  — absolute path to sparse_text dir
        "zip"      — absolute path to Colab upload ZIP
    """
    import sys
    _repo = Path(__file__).resolve().parents[1]
    for _p in (str(_repo / "src"), str(_repo)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    video_path  = Path(video_path)
    output_root = Path(output_root)

    job_dir     = output_root / video_path.stem
    frames_dir  = job_dir / "frames"
    colmap_dir  = job_dir / "colmap"
    sparse_text = colmap_dir / "sparse_text"

    for d in (frames_dir, colmap_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Step 1: Frame extraction ──────────────────────────────────────────
    log.info("Step 1: Frame extraction")
    from preprocessing.extract_frames import extract_from_video
    n = extract_from_video(
        video_path=str(video_path),
        output_dir=str(frames_dir),
        fps=fps,
        max_frames=max_frames,
    )
    log.info("  Extracted %d frames → %s", n, frames_dir)

    # ── Step 2: Smart frame selection ────────────────────────────────────
    try:
        from preprocessing.extract_frames import run_smart_frame_selection
        report = run_smart_frame_selection(str(frames_dir), budget=max_frames)
        log.info(
            "  Smart selection: %d/%d kept",
            report["selected_frame_count"],
            report["original_frame_count"],
        )
    except Exception as exc:
        log.warning("  Smart frame selection skipped: %s", exc)

    # ── Step 3 (optional): Quality gate via core/ ─────────────────────────
    if use_quality_gate:
        _run_quality_gate(frames_dir, job_dir)

    # ── Step 4: COLMAP ───────────────────────────────────────────────────
    log.info("Step 4: COLMAP sparse reconstruction")
    from preprocessing.colmap_runner import run_colmap
    stats = run_colmap(
        image_dir=str(frames_dir),
        output_dir=str(colmap_dir),
        quality=quality,
        use_gpu=use_gpu,
    )
    log.info(
        "  COLMAP: %d/%d registered, %s points",
        stats.get("registered", 0),
        stats.get("total", 0),
        f"{stats.get('n_points', 0):,}",
    )

    if not (sparse_text / "cameras.txt").exists():
        raise RuntimeError(
            f"COLMAP did not produce cameras.txt in {sparse_text}. "
            "Check COLMAP logs above."
        )

    # ── Step 5: Package for Colab ─────────────────────────────────────────
    log.info("Step 5: Packaging for Colab")
    zip_path = _build_colab_zip(job_dir, frames_dir, sparse_text)
    log.info("  ZIP ready: %s  (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    elapsed = time.time() - t0
    log.info("Pipeline complete in %.0fs", elapsed)

    return {
        "frames":  str(frames_dir.resolve()),
        "dataset": str(sparse_text.resolve()),
        "zip":     str(zip_path.resolve()),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_quality_gate(frames_dir: Path, job_dir: Path) -> None:
    """
    Optional core/ quality gate — runs dataset analysis + success prediction
    and logs a warning if the dataset is high-risk.  Does NOT abort the pipeline;
    the caller can inspect job_dir/quality_report.json to decide.
    """
    log.info("Step 3: Dataset quality gate (core/)")
    try:
        from core.dataset_analysis import DatasetAnalysisPipeline
        from core.quality_prediction import ReconstructionSuccessPredictor

        qr = DatasetAnalysisPipeline().analyze(
            frames_dir, output_path=job_dir / "quality_report.json"
        )
        pr = ReconstructionSuccessPredictor().predict(
            qr, output_path=job_dir / "prediction_report.json"
        )
        log.info(
            "  Quality: blur=%.2f coverage=%.2f texture=%.2f success_prob=%.2f",
            qr.get("blur_score", 0),
            qr.get("coverage_score", 0),
            qr.get("texture_score", 0),
            qr.get("success_probability", 0),
        )
        if pr.get("risk_level") == "high":
            log.warning("  HIGH-RISK dataset — %s", pr.get("recommended_action", ""))
            for reason in pr.get("explanation", {}).get("risk_factors", []):
                log.warning("    • %s", reason)
    except Exception as exc:
        log.warning("  Quality gate failed (non-fatal): %s", exc)


def _build_colab_zip(job_dir: Path, frames_dir: Path, sparse_text: Path) -> Path:
    """Build the Colab training ZIP from frames + COLMAP sparse_text."""
    zip_path = job_dir / "colab_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in sorted(sparse_text.rglob("*")):
            if f.is_file():
                zf.write(f, Path("sparse_text") / f.name)
        for f in sorted(frames_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                zf.write(f, Path("frames") / f.name)
    return zip_path
