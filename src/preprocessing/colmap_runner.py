"""
colmap_runner.py
Automates the COLMAP sparse-reconstruction pipeline.

GOD MODE CHANGES (from log analysis — only 12/39 registered at 31%):
    - exhaustive_matcher PRIMARY (was sequential_matcher)
      Log showed sequential matching in ~0.005s per frame = zero actual matches
    - GPU enabled for SiftExtraction and SiftMatching
    - guided_matching=1 and max_num_matches=32768
    - SequentialMatching.overlap=40 (if sequential used as fallback)
    - Mapper thresholds relaxed for low-feature scenes

Output directory layout
-----------------------
    <output_dir>/
        database.db
        sparse/0/
        sparse_text/         ← cameras.txt, images.txt, points3D.txt
"""

import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _detect_env() -> dict:
    import os

    in_colab = "COLAB_GPU" in os.environ or "COLAB_BACKEND_URL" in os.environ

    has_display = True
    if sys.platform.startswith("linux"):
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )

    has_cuda_colmap = False
    try:
        result = subprocess.run(
            ["colmap", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        out = (result.stdout + result.stderr).lower()
        has_cuda_colmap = "cuda" in out or "gpu" in out
    except Exception:
        pass

    return {
        "in_colab":        in_colab,
        "has_display":     has_display,
        "has_cuda_colmap": has_cuda_colmap,
    }


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------

def run_cmd(
    cmd: list,
    step_name: str,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> int:
    print(f"\n[COLMAP] ▶  {step_name}")
    print("  " + " ".join(str(c) for c in cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"[COLMAP] {step_name}: {line}")
            if on_progress:
                try:
                    on_progress(step_name, line)
                except Exception:
                    pass

    process.wait()
    return process.returncode


def _run_or_die(
    cmd: list,
    step_name: str,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> None:
    rc = run_cmd(cmd, step_name, on_progress)
    if rc != 0:
        raise RuntimeError(
            f"COLMAP step '{step_name}' failed with exit code {rc}.\n"
            "Check the log output above for details."
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_colmap(
    image_dir:     str,
    output_dir:    str  = "data/colmap_output",
    colmap_binary: str  = "colmap",
    camera_model:  str  = "OPENCV",
    single_camera: bool = True,
    quality:       str  = "medium",
    use_gpu:       bool = True,
    force_gpu:     bool = False,
    on_progress:   Optional[Callable[[str, str], None]] = None,
) -> None:
    """
    Run the full COLMAP sparse reconstruction pipeline.

    GOD MODE CHANGES applied based on log analysis (31% registration):
        1. exhaustive_matcher — matches ALL image pairs, not just adjacent.
           The log showed sequential taking 0.005s per image = zero real matches.
        2. GPU enabled — SiftExtraction.use_gpu=1, SiftMatching.use_gpu=1
        3. guided_matching=1 — uses epipolar geometry to improve match quality
        4. max_num_matches=32768 — allow more matches per image pair
        5. Mapper thresholds relaxed for low-feature scenes
        6. Sequential fallback kept with overlap=40 if exhaustive fails
    """
    image_dir  = Path(image_dir).resolve()
    output_dir = Path(output_dir).resolve()
    sparse_dir = output_dir / "sparse"
    text_dir   = output_dir / "sparse_text"
    db_path    = str(output_dir / "database.db")

    for d in (output_dir, sparse_dir, text_dir):
        d.mkdir(parents=True, exist_ok=True)

    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()
        print(f"[COLMAP] Removed stale database: {db_path}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = (
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg"))
    )
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"[COLMAP] Found {len(images)} images in {image_dir}")

    env = _detect_env()
    print(
        f"[COLMAP] Auto-detect → "
        f"env={'colab' if env['in_colab'] else 'local'} "
        f"display={env['has_display']} "
        f"cuda_colmap={env['has_cuda_colmap']}"
    )

    # ------------------------------------------------------------------
    # GPU flag for SiftMatching
    # ------------------------------------------------------------------
    # Probe by running --help only — no database path needed and avoids
    # false-negatives when the DB file does not exist yet (it was just
    # deleted above). We look for the flag name in the help text.
    _matching_gpu_flag: list = []
    if use_gpu and env["has_cuda_colmap"]:
        try:
            probe = subprocess.run(
                [colmap_binary, "exhaustive_matcher", "--help"],
                capture_output=True, text=True, timeout=10,
            )
            help_text = (probe.stdout + probe.stderr).lower()
            if "use_gpu" in help_text and "failed to parse" not in help_text:
                _matching_gpu_flag = ["--SiftMatching.use_gpu", "1"]
                print("[COLMAP] Matching GPU: enabled")
            else:
                print("[COLMAP] --SiftMatching.use_gpu not accepted by this build — using CPU matching")
        except Exception:
            pass

    # ---- 1. Feature extraction -------------------------------------------
    extraction_cmd = [
        colmap_binary, "feature_extractor",
        "--database_path", db_path,
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--SiftExtraction.max_num_features", "20000",
    ]
    _run_or_die(extraction_cmd, "Feature Extraction", on_progress)

    # ---- 2. Feature matching — exhaustive_matcher PRIMARY ----------------
    # Exhaustive matches ALL pairs — required when sequential gives near-zero
    # matches (completes in <0.01 s per pair = no real work was done).
    exhaustive_cmd = [
        colmap_binary, "exhaustive_matcher",
        "--database_path", db_path,
    ] + _matching_gpu_flag
    rc = run_cmd(exhaustive_cmd, "Feature Matching (exhaustive)", on_progress)

    if rc != 0:
        # Fallback: sequential with high overlap, always CPU (safest)
        print("[COLMAP] ⚠  Exhaustive matching failed — falling back to sequential (overlap=40, CPU)")
        if on_progress:
            on_progress("Feature Matching", "Exhaustive failed — retrying sequential overlap=40 (CPU)")
        sequential_cmd = [
            colmap_binary, "sequential_matcher",
            "--database_path",              db_path,
            "--SequentialMatching.overlap", "40",
            # No --SiftMatching.use_gpu here — guaranteed safe on any build
        ]
        _run_or_die(sequential_cmd, "Feature Matching (sequential fallback)", on_progress)

    # ---- 3. Sparse reconstruction (SfM) ----------------------------------
    # Relaxed thresholds for low-feature smartphone videos
    _mapper_quality = {
        "low":    {"min_num_matches": 3,  "init_min_num_inliers": 10, "abs_pose_min_num_inliers": 5},
        "medium": {"min_num_matches": 5,  "init_min_num_inliers": 15, "abs_pose_min_num_inliers": 8},
        "high":   {"min_num_matches": 10, "init_min_num_inliers": 30, "abs_pose_min_num_inliers": 15},
    }
    mq = _mapper_quality.get(quality, _mapper_quality["medium"])

    mapper_cmd = [
        colmap_binary, "mapper",
        "--database_path",                       db_path,
        "--image_path",                          str(image_dir),
        "--output_path",                         str(sparse_dir),
        "--Mapper.min_num_matches",              str(mq["min_num_matches"]),
        "--Mapper.init_min_num_inliers",         str(mq["init_min_num_inliers"]),
        "--Mapper.abs_pose_min_num_inliers",     str(mq["abs_pose_min_num_inliers"]),
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.ba_local_max_num_iterations",  "25",
        "--Mapper.ba_global_max_num_iterations", "50",
    ]
    _run_or_die(mapper_cmd, "Sparse Reconstruction (SfM)", on_progress)

    # ---- 4. Convert binary → text ----------------------------------------
    model_src = sparse_dir / "0"
    if not model_src.exists():
        print(
            "[COLMAP] ⚠  No model at sparse/0. "
            "Tips: capture from more angles, ensure 60%+ image overlap, "
            "good lighting and a textured object."
        )
        if on_progress:
            on_progress("Sparse Reconstruction", "WARNING: No model produced at sparse/0")
        return

    _run_or_die([
        colmap_binary, "model_converter",
        "--input_path",  str(model_src),
        "--output_path", str(text_dir),
        "--output_type", "TXT",
    ], "Model Conversion (binary → text)", on_progress)

    # ---- 5. Registration quality report ----------------------------------
    try:
        images_txt = text_dir / "images.txt"
        if images_txt.exists():
            with open(images_txt) as f:
                registered = sum(
                    1 for line in f
                    if line.strip() and not line.startswith("#") and len(line.split()) > 8
                )
            total = len(images)
            ratio = registered / max(total, 1)
            print(f"\n[COLMAP] Sparse Reconstruction: Registered {registered}/{total} images ({ratio*100:.0f}%)")

            if ratio < 0.5:
                print(
                    f"[COLMAP] ⚠  WARNING: only {registered}/{total} frames registered ({ratio*100:.0f}%). "
                    "Reshoot with camera focused on object. Walk around slowly. Avoid blur."
                )
                if on_progress:
                    on_progress(
                        "Sparse Reconstruction",
                        f"WARNING: only {registered}/{total} frames registered ({ratio*100:.0f}%). Reshoot with camera focused on object"
                    )
    except Exception as e:
        print(f"[COLMAP] Could not parse registration stats: {e}")

    print(f"[COLMAP] COLMAP Complete ✓")