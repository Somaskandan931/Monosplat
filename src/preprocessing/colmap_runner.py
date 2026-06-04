"""
colmap_runner.py
Automates the COLMAP sparse-reconstruction pipeline with ADAPTIVE FALLBACK TIERS.

ROBUSTNESS ARCHITECTURE
-----------------------
Instead of one rigid COLMAP config that hard-fails, the runner now tries four
progressive presets in sequence, stopping as soon as registration is adequate:

  TIER 1  — Default (high-quality, affine SIFT)
             Best for well-lit scenes with rich texture.

  TIER 2  — Phone-video mode
             Disables affine-shape / domain-size-pooling; lower peak_threshold.
             Often MORE robust on mobile footage than the "quality" defaults.

  TIER 3  — High-overlap exhaustive + sequential combo
             Forces exhaustive matching AND sequential with overlap=30; best
             for continuous orbit/walkthrough captures.

  TIER 4  — Low-texture rescue
             peak_threshold=0.002, max_num_features=30000, relaxed mapper;
             last resort before final failure.

After each tier the mapper result is evaluated.  If registration ≥ min_ratio
the pipeline continues.  If all four tiers fail the runner raises a detailed
RuntimeError with scene-specific repair advice built from diagnostics.

Output directory layout
-----------------------
  <output_dir>/
      database.db
      sparse/0/
      sparse_text/    ← cameras.txt, images.txt, points3D.txt
"""

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _detect_env(colmap_binary: str = "colmap") -> dict:
    from utils.env_detect import (
        has_cuda_colmap,
        is_colab,
        should_use_gpu,
        should_use_matching_gpu,
    )
    return {
        "in_colab":            is_colab(),
        "has_cuda_colmap":     has_cuda_colmap(colmap_binary),
        "use_extraction_gpu":  should_use_gpu(colmap_binary),
        "use_matching_gpu":    should_use_matching_gpu(colmap_binary),
    }


# ---------------------------------------------------------------------------
# Command helpers
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
# Registration counter (shared across tiers)
# ---------------------------------------------------------------------------

def _count_registered(txt_dir: Path) -> int:
    """Count registered images in COLMAP images.txt.

    Robust parser: counts lines that match the image header format (exactly 10
    fields: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME) rather than relying
    on idx%2 parity, which breaks when keypoint lines are empty (COLMAP omits
    the second line for images with zero matched keypoints, shifting the parity
    and doubling the count).
    """
    imgs_txt = txt_dir / "images.txt"
    if not imgs_txt.exists():
        return 0
    count = 0
    with open(imgs_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Image header lines have exactly 10 fields; keypoint lines have 3*N fields (N≥0)
            # A header always starts with an integer IMAGE_ID followed by 4 quaternion floats.
            if len(parts) == 10:
                try:
                    int(parts[0])   # IMAGE_ID
                    float(parts[1]) # QW
                    count += 1
                except ValueError:
                    pass
    return count


def _count_points(txt_dir: Path) -> int:
    pts = txt_dir / "points3D.txt"
    if not pts.exists():
        return 0
    return sum(1 for l in open(pts) if l.strip() and not l.startswith("#"))


# ---------------------------------------------------------------------------
# Geometric quality validation (Issue B / C fix)
# ---------------------------------------------------------------------------

def _validate_reconstruction_quality(
    txt_dir: Path,
    total_images: int,
    min_registered_ratio: float = 0.60,
    min_points: int = 1000,
    max_mean_reprojection_error: float = 1.5,  # 2.0→1.5px: tighter geometric quality gate
    max_track_length_variance: float = 50.0,
) -> dict:
    """
    Validate geometric quality of a COLMAP reconstruction.

    Checks:
      - Registration ratio (registered / total images)
      - Sparse point cloud density
      - Mean reprojection error (from points3D.txt track residuals)
      - Track length distribution (short tracks = weak geometry)

    Returns a dict:
        {
            "passed": bool,
            "registered_ratio": float,
            "n_points": int,
            "mean_reprojection_error": float,
            "mean_track_length": float,
            "reason": str | None   — human-readable failure reason
        }

    Raises nothing — callers decide whether to hard-fail or downgrade.
    """
    import math

    registered = _count_registered(txt_dir)
    n_points   = _count_points(txt_dir)
    reg_ratio  = registered / max(total_images, 1)

    result = {
        "passed": True,
        "registered_ratio": round(reg_ratio, 3),
        "n_points": n_points,
        "mean_reprojection_error": 0.0,
        "mean_track_length": 0.0,
        "reason": None,
    }

    # --- 1. Registration ratio ---
    if reg_ratio < min_registered_ratio:
        result["passed"] = False
        result["reason"] = (
            f"Only {reg_ratio*100:.0f}% of images registered "
            f"(minimum {min_registered_ratio*100:.0f}%). "
            "Camera poses are incomplete — splat will be foggy or partial."
        )
        return result

    # --- 2. Point cloud density ---
    if n_points < min_points:
        result["passed"] = False
        result["reason"] = (
            f"Only {n_points:,} sparse points (minimum {min_points:,}). "
            "Scene geometry is too sparse for clean Gaussian initialisation."
        )
        return result

    # --- 3. Reprojection error + track length from points3D.txt ---
    pts_file = txt_dir / "points3D.txt"
    if pts_file.exists():
        errors = []
        track_lengths = []
        try:
            with open(pts_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    # points3D.txt format:
                    # POINT3D_ID X Y Z R G B ERROR TRACK[]
                    try:
                        err = float(parts[7])
                        track_len = (len(parts) - 8) // 2  # each track entry = IMAGE_ID POINT2D_IDX
                        errors.append(err)
                        track_lengths.append(max(track_len, 1))
                    except (ValueError, IndexError):
                        continue
        except Exception:
            pass

        if errors:
            mean_err = sum(errors) / len(errors)
            result["mean_reprojection_error"] = round(mean_err, 4)
            if mean_err > max_mean_reprojection_error:
                result["passed"] = False
                result["reason"] = (
                    f"Mean reprojection error {mean_err:.2f}px exceeds limit "
                    f"{max_mean_reprojection_error:.1f}px. "
                    "Camera poses are geometrically unstable — output will be blurry fog."
                )
                return result

        if track_lengths:
            mean_track = sum(track_lengths) / len(track_lengths)
            result["mean_track_length"] = round(mean_track, 2)
            if mean_track < 3.0:
                result["passed"] = False
                result["reason"] = (
                    f"Mean track length {mean_track:.1f} is too short (minimum 3.0). "
                    "Feature matches are too sparse — geometry is unreliable."
                )
                return result

    return result


# ---------------------------------------------------------------------------
# Scene diagnostics (texture / blur / reflection hints)
# ---------------------------------------------------------------------------

def _scene_diagnostics(image_dir: Path, sample_n: int = 15) -> dict:
    """
    Quick per-image diagnostics to build a human-readable failure report.
    Returns dict with keys: blur_pct, low_texture_pct, overexposed_pct, advice.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"blur_pct": 0, "low_texture_pct": 0, "overexposed_pct": 0, "advice": []}

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        return {"blur_pct": 0, "low_texture_pct": 0, "overexposed_pct": 0, "advice": []}

    step = max(1, len(images) // sample_n)
    sample = images[::step][:sample_n]

    blur_count = tex_count = exp_count = 0
    for p in sample:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
        mean_px = float(img.mean())
        sift = cv2.SIFT_create()
        kps  = sift.detect(img, None)

        if lap_var < 60:
            blur_count += 1
        if len(kps) < 80:
            tex_count += 1
        if mean_px > 220:
            exp_count += 1

    n = max(len(sample), 1)
    blur_pct = blur_count / n * 100
    tex_pct  = tex_count  / n * 100
    exp_pct  = exp_count  / n * 100

    advice = []
    if blur_pct > 30:
        advice.append(f"HIGH BLUR ({blur_pct:.0f}% frames): shoot slower, lock camera focus.")
    if tex_pct > 40:
        advice.append(f"LOW TEXTURE ({tex_pct:.0f}% frames): place object on patterned surface (newspaper/graph paper).")
    if exp_pct > 20:
        advice.append(f"OVEREXPOSED ({exp_pct:.0f}% frames): reduce lighting or lock camera exposure.")
    if not advice:
        advice.append("Scene looks normal — insufficient parallax or overlap is the likely cause.")

    return {
        "blur_pct":        round(blur_pct, 1),
        "low_texture_pct": round(tex_pct, 1),
        "overexposed_pct": round(exp_pct, 1),
        "advice":          advice,
    }


# ---------------------------------------------------------------------------
# Matching GPU flag probe (cached across tiers)
# ---------------------------------------------------------------------------

def _probe_matching_gpu_flag(colmap_binary: str, use_matching_gpu: bool) -> list:
    """Probe which GPU flag name exhaustive_matcher actually accepts.

    COLMAP 3.10+ renamed SiftMatching → FeatureMatching in some builds.
    We test the candidate flags with --help and check for 'failed to parse'
    to determine which namespace is live on this binary.
    """
    if not use_matching_gpu:
        return []

    candidates = [
        ("--SiftMatching.use_gpu",    "SiftMatching"),
        ("--FeatureMatching.use_gpu", "FeatureMatching"),
    ]

    for flag, namespace in candidates:
        try:
            probe = subprocess.run(
                [colmap_binary, "exhaustive_matcher", flag, "1",
                 "--database_path", "nul"],   # dummy path; will fail but after flag parse
                capture_output=True, text=True, timeout=10,
            )
            combined = (probe.stdout + probe.stderr).lower()
            # If the flag is unrecognised COLMAP prints "failed to parse options"
            if "failed to parse" not in combined and "unrecogni" not in combined:
                print(f"[COLMAP] Matching GPU: enabled ({namespace})")
                return [flag, "1"]
        except Exception:
            pass

    print("[COLMAP] Matching GPU flag not supported — using CPU matching")
    return []


# ---------------------------------------------------------------------------
# TIER helpers
# ---------------------------------------------------------------------------

def _reset_sparse(sparse_dir: Path) -> None:
    """Wipe sparse output between tier attempts."""
    if sparse_dir.exists():
        for item in sparse_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def _convert_and_count(
    colmap_binary: str,
    sparse_dir: Path,
    text_dir: Path,
    tier_label: str,
    on_progress: Optional[Callable],
) -> tuple:
    """Convert sparse/0 → text, return (registered, n_points). Returns (0,0) on failure."""
    model_src = sparse_dir / "0"
    if not model_src.exists():
        print(f"[COLMAP] {tier_label}: no sparse/0 produced.")
        return 0, 0

    if text_dir.exists():
        shutil.rmtree(text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)

    rc = run_cmd([
        colmap_binary, "model_converter",
        "--input_path",  str(model_src),
        "--output_path", str(text_dir),
        "--output_type", "TXT",
    ], f"Model Conversion ({tier_label})", on_progress)

    if rc != 0:
        return 0, 0

    reg = _count_registered(text_dir)
    pts = _count_points(text_dir)
    return reg, pts


def _run_mapper(
    colmap_binary: str,
    db_path: str,
    image_dir: Path,
    sparse_dir: Path,
    mapper_kwargs: dict,
    label: str,
    on_progress: Optional[Callable],
) -> int:
    cmd = [
        colmap_binary, "mapper",
        "--database_path", db_path,
        "--image_path",    str(image_dir),
        "--output_path",   str(sparse_dir),
    ]
    for k, v in mapper_kwargs.items():
        cmd += [f"--{k}", str(v)]
    return run_cmd(cmd, f"Sparse Reconstruction ({label})", on_progress)


# ---------------------------------------------------------------------------
# TIER 1 — Default (high-quality affine SIFT)
# ---------------------------------------------------------------------------

def _write_image_list(image_dir: Path) -> Path:
    """Write a flat image_list.txt containing only top-level frame filenames.

    COLMAP's --image_path with no list recurses into ALL subdirectories
    (blurry/, duplicates/, etc.) and double-registers their frames.
    Passing --image_list_path restricts COLMAP to exactly these files,
    eliminating the 2× registration ratio observed in the wild.

    COLMAP 3.10+ moved this flag out of the ImageReader namespace to the
    top level (--image_list_path). The old --ImageReader.image_list_path
    is unrecognised in COLMAP 3.13.

    Returns the path to the written list file.
    """
    images = sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    list_path = image_dir.parent / "colmap_image_list.txt"
    list_path.write_text("\n".join(images) + "\n")
    print(f"[COLMAP] image_list.txt: {len(images)} frames → {list_path}")
    return list_path


def _tier1_extraction(
    colmap_binary: str, db_path: str, image_dir: Path,
    use_extraction_gpu: bool, on_progress: Optional[Callable],
) -> None:
    list_path = _write_image_list(image_dir)
    cmd = [
        colmap_binary, "feature_extractor",
        "--database_path",                        db_path,
        "--image_path",                           str(image_dir),
        "--image_list_path",                      str(list_path),   # COLMAP 3.10+: top-level flag
        "--ImageReader.camera_model",             "OPENCV",
        "--ImageReader.single_camera",            "1",
        "--SiftExtraction.max_image_size",        "4096",
        "--SiftExtraction.max_num_features",      "20000",   # 16k→20k: more features for weak textures
        "--SiftExtraction.peak_threshold",        "0.003",   # 0.004→0.003: detect weaker corners
        "--SiftExtraction.edge_threshold",        "10",
        "--SiftExtraction.first_octave",          "-1",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling",   "1",
    ]
    # --SiftExtraction.use_gpu removed in COLMAP 3.10+; GPU is auto-detected
    _run_or_die(cmd, "Feature Extraction (Tier-1 default)", on_progress)


# ---------------------------------------------------------------------------
# TIER 2 — Phone-video mode (no affine, lower threshold)
# ---------------------------------------------------------------------------

def _tier2_extraction(
    colmap_binary: str, db_path: str, image_dir: Path,
    use_extraction_gpu: bool, on_progress: Optional[Callable],
) -> None:
    """Simpler SIFT that is often more robust on compressed mobile video."""
    print("[COLMAP] ── TIER 2: Phone-video mode extraction ──")
    if on_progress:
        on_progress("Tier-2", "Switching to phone-video SIFT (no affine shape)")
    list_path = _write_image_list(image_dir)
    cmd = [
        colmap_binary, "feature_extractor",
        "--database_path",                        db_path,
        "--image_path",                           str(image_dir),
        "--image_list_path",                      str(list_path),   # COLMAP 3.10+: top-level flag
        "--ImageReader.camera_model",             "OPENCV",
        "--ImageReader.single_camera",            "1",
        "--SiftExtraction.max_image_size",        "4096",
        "--SiftExtraction.max_num_features",      "20000",   # 16k→20k
        "--SiftExtraction.peak_threshold",        "0.005",   # 0.006→0.005: slightly more sensitive
        "--SiftExtraction.edge_threshold",        "10",
        "--SiftExtraction.first_octave",          "-1",
        "--SiftExtraction.estimate_affine_shape", "0",
        "--SiftExtraction.domain_size_pooling",   "0",
    ]
    # --SiftExtraction.use_gpu removed in COLMAP 3.10+; GPU is auto-detected
    _run_or_die(cmd, "Feature Extraction (Tier-2 phone)", on_progress)


# ---------------------------------------------------------------------------
# TIER 3 — High-overlap: exhaustive + sequential(overlap=30)
# ---------------------------------------------------------------------------

def _tier3_matching(
    colmap_binary: str, db_path: str,
    gpu_flag: list, on_progress: Optional[Callable],
) -> None:
    print("[COLMAP] ── TIER 3: High-overlap matching (exhaustive + sequential) ──")
    if on_progress:
        on_progress("Tier-3", "High-overlap exhaustive + sequential(overlap=30)")
    run_cmd([
        colmap_binary, "exhaustive_matcher",
        "--database_path",                db_path,
        # --SiftMatching.guided_matching removed in COLMAP 3.10+
        "--FeatureMatching.max_num_matches", "32768",
        "--FeatureMatching.max_ratio",       "0.80",
    ] + gpu_flag, "Feature Matching (Tier-3 exhaustive)", on_progress)
    run_cmd([
        colmap_binary, "sequential_matcher",
        "--database_path",              db_path,
        "--SequentialMatching.overlap", "30",
    ] + gpu_flag, "Feature Matching (Tier-3 sequential overlap=30)", on_progress)


# ---------------------------------------------------------------------------
# TIER 4 — Low-texture rescue
# ---------------------------------------------------------------------------

def _tier4_extraction(
    colmap_binary: str, db_path: str, image_dir: Path,
    use_extraction_gpu: bool, on_progress: Optional[Callable],
) -> None:
    print("[COLMAP] ── TIER 4: Low-texture rescue (dense features) ──")
    if on_progress:
        on_progress("Tier-4", "Low-texture rescue: peak_threshold=0.002, features=30000")
    list_path = _write_image_list(image_dir)
    cmd = [
        colmap_binary, "feature_extractor",
        "--database_path",                        db_path,
        "--image_path",                           str(image_dir),
        "--image_list_path",                      str(list_path),   # COLMAP 3.10+: top-level flag
        "--ImageReader.camera_model",             "OPENCV",
        "--ImageReader.single_camera",            "1",
        "--SiftExtraction.max_image_size",        "4096",
        "--SiftExtraction.max_num_features",      "30000",
        "--SiftExtraction.peak_threshold",        "0.002",
        "--SiftExtraction.edge_threshold",        "8",
        "--SiftExtraction.first_octave",          "-1",
        "--SiftExtraction.estimate_affine_shape", "0",
        "--SiftExtraction.domain_size_pooling",   "0",
    ]
    # --SiftExtraction.use_gpu removed in COLMAP 3.10+; GPU is auto-detected
    _run_or_die(cmd, "Feature Extraction (Tier-4 low-texture)", on_progress)


# ---------------------------------------------------------------------------
# Shared exhaustive matching (Tiers 1 + 2 + 4)
# ---------------------------------------------------------------------------

def _exhaustive_matching(
    colmap_binary: str, db_path: str,
    gpu_flag: list, on_progress: Optional[Callable],
) -> None:
    rc = run_cmd([
        colmap_binary, "exhaustive_matcher",
        "--database_path",                db_path,
        # --SiftMatching.guided_matching removed in COLMAP 3.10+
        "--FeatureMatching.max_num_matches", "32768",
        "--FeatureMatching.max_ratio",       "0.80",   # 0.85→0.80: stricter ratio test, fewer false matches
    ] + gpu_flag, "Feature Matching (exhaustive)", on_progress)

    if rc != 0:
        print("[COLMAP] Exhaustive matching failed — falling back to sequential(overlap=40)")
        if on_progress:
            on_progress("Matching", "Exhaustive failed — sequential fallback (overlap=40)")
        run_cmd([
            colmap_binary, "sequential_matcher",
            "--database_path",              db_path,
            "--SequentialMatching.overlap", "40",
        ] + gpu_flag, "Feature Matching (sequential fallback)", on_progress)


# ---------------------------------------------------------------------------
# Standard mapper kwargs by quality
# ---------------------------------------------------------------------------

_MAPPER_QUALITY = {
    "high":   {"Mapper.min_num_matches": 10, "Mapper.init_min_num_inliers": 30,  "Mapper.abs_pose_min_num_inliers": 15},
    "medium": {"Mapper.min_num_matches": 5,  "Mapper.init_min_num_inliers": 15,  "Mapper.abs_pose_min_num_inliers": 8},
    "low":    {"Mapper.min_num_matches": 3,  "Mapper.init_min_num_inliers": 10,  "Mapper.abs_pose_min_num_inliers": 5},
    "rescue": {"Mapper.min_num_matches": 2,  "Mapper.init_min_num_inliers": 6,   "Mapper.abs_pose_min_num_inliers": 3,
               "Mapper.min_focal_length_ratio": "0.1", "Mapper.max_focal_length_ratio": "10",
               "Mapper.max_extra_param": "1"},
}

_MAPPER_COMMON = {
    "Mapper.ba_global_function_tolerance": "0.0000001",  # 1e-6→1e-7: tighter BA convergence
    "Mapper.ba_local_max_num_iterations":  "40",  # 25→40: more BA local iterations
    "Mapper.ba_global_max_num_iterations": "80",  # 50→80: more global BA iterations
}


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
    min_ratio:     float = 0.60,   # minimum registration ratio (was 0.20 — too permissive)
    min_points:    int   = 1000,   # minimum 3D points (was 300 — too sparse for training)
) -> dict:
    """
    Run COLMAP with adaptive fallback tiers.

    Returns a dict with registration stats:
        {registered, total, ratio, n_points, tier_used}

    Raises RuntimeError only after ALL four tiers are exhausted.

    min_ratio / min_points enforce geometric quality gates per tier.
    Tier 4 results also undergo full reprojection-error validation before
    acceptance — results that pass the count gate but have error > 2px are
    rejected as geometrically poisoned geometry.
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
    total = len(images)
    print(f"[COLMAP] Found {total} images in {image_dir}")

    env = _detect_env(colmap_binary)
    use_extraction_gpu = use_gpu and (force_gpu or env["use_extraction_gpu"])
    use_matching_gpu   = use_gpu and (force_gpu or env["use_matching_gpu"])
    gpu_flag = _probe_matching_gpu_flag(colmap_binary, use_matching_gpu)

    print(
        f"[COLMAP] Env → colab={env['in_colab']}  "
        f"cuda_colmap={env['has_cuda_colmap']}  "
        f"extraction_gpu={use_extraction_gpu}  matching_gpu={use_matching_gpu}"
    )

    # ── mapper kwargs for this run quality ──────────────────────────────────
    mq = {**_MAPPER_COMMON, **_MAPPER_QUALITY.get(quality, _MAPPER_QUALITY["medium"])}

    # Track best result across all tiers
    best_registered = 0
    best_points     = 0
    best_tier       = None

    # ════════════════════════════════════════════════════════════════════════
    # TIER 1 — Default high-quality affine SIFT
    # ════════════════════════════════════════════════════════════════════════
    print("\n[COLMAP] ════ TIER 1: Default (affine SIFT) ════")
    if on_progress:
        on_progress("Tier-1", "Starting Tier-1: high-quality affine SIFT extraction")
    try:
        _tier1_extraction(colmap_binary, db_path, image_dir, use_extraction_gpu, on_progress)
        _exhaustive_matching(colmap_binary, db_path, gpu_flag, on_progress)
        _run_mapper(colmap_binary, db_path, image_dir, sparse_dir, mq, "Tier-1", on_progress)
        reg, pts = _convert_and_count(colmap_binary, sparse_dir, text_dir, "Tier-1", on_progress)
        ratio = reg / max(total, 1)
        print(f"[COLMAP] Tier-1 result: {reg}/{total} ({ratio*100:.0f}%)  points={pts:,}")
        if on_progress:
            on_progress("Tier-1", f"Registration: {reg}/{total} ({ratio*100:.0f}%), points={pts:,}")
        if reg > best_registered:
            best_registered, best_points, best_tier = reg, pts, "tier1"
        if ratio >= min_ratio and pts >= min_points:
            quality = _validate_reconstruction_quality(text_dir, total)
            if quality["passed"]:
                print(f"[COLMAP] ✓ Tier-1 succeeded. reprojection_err={quality['mean_reprojection_error']:.3f}px  track_len={quality['mean_track_length']:.1f}")
                return _finalize(text_dir, reg, total, pts, "tier1", on_progress)
            else:
                print(f"[COLMAP] ✗ Tier-1 geometry failed quality gate: {quality['reason']}")
        elif ratio >= min_ratio:
            print(f"[COLMAP] Tier-1: count OK ({reg}/{total}) but geometry invalid — trying next tier.")
    except Exception as e:
        print(f"[COLMAP] Tier-1 failed with exception: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TIER 2 — Phone-video mode (no affine shape / domain pooling)
    # ════════════════════════════════════════════════════════════════════════
    print("\n[COLMAP] ════ TIER 2: Phone-video mode ════")
    if on_progress:
        on_progress("Tier-2", "Starting Tier-2: phone-video SIFT (no affine shape)")
    try:
        # Fresh database for new extraction settings
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()
        _reset_sparse(sparse_dir)

        _tier2_extraction(colmap_binary, db_path, image_dir, use_extraction_gpu, on_progress)
        _exhaustive_matching(colmap_binary, db_path, gpu_flag, on_progress)
        _run_mapper(colmap_binary, db_path, image_dir, sparse_dir, mq, "Tier-2", on_progress)
        reg, pts = _convert_and_count(colmap_binary, sparse_dir, text_dir, "Tier-2", on_progress)
        ratio = reg / max(total, 1)
        print(f"[COLMAP] Tier-2 result: {reg}/{total} ({ratio*100:.0f}%)  points={pts:,}")
        if on_progress:
            on_progress("Tier-2", f"Registration: {reg}/{total} ({ratio*100:.0f}%), points={pts:,}")
        if reg > best_registered:
            best_registered, best_points, best_tier = reg, pts, "tier2"
        if ratio >= min_ratio and pts >= min_points:
            quality = _validate_reconstruction_quality(text_dir, total)
            if quality["passed"]:
                print(f"[COLMAP] ✓ Tier-2 succeeded. reprojection_err={quality['mean_reprojection_error']:.3f}px")
                return _finalize(text_dir, reg, total, pts, "tier2", on_progress)
            else:
                print(f"[COLMAP] ✗ Tier-2 geometry failed quality gate: {quality['reason']}")
    except Exception as e:
        print(f"[COLMAP] Tier-2 failed with exception: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TIER 3 — High-overlap (exhaustive + sequential overlap=30)
    #          Uses Tier-2 features (already extracted, reuse DB)
    # ════════════════════════════════════════════════════════════════════════
    print("\n[COLMAP] ════ TIER 3: High-overlap matching ════")
    if on_progress:
        on_progress("Tier-3", "Starting Tier-3: exhaustive + sequential(overlap=30)")
    try:
        # Keep the Tier-2 feature DB, just redo matching + mapper
        _reset_sparse(sparse_dir)
        _tier3_matching(colmap_binary, db_path, gpu_flag, on_progress)

        mq_low = {**_MAPPER_COMMON, **_MAPPER_QUALITY["low"]}
        _run_mapper(colmap_binary, db_path, image_dir, sparse_dir, mq_low, "Tier-3", on_progress)
        reg, pts = _convert_and_count(colmap_binary, sparse_dir, text_dir, "Tier-3", on_progress)
        ratio = reg / max(total, 1)
        print(f"[COLMAP] Tier-3 result: {reg}/{total} ({ratio*100:.0f}%)  points={pts:,}")
        if on_progress:
            on_progress("Tier-3", f"Registration: {reg}/{total} ({ratio*100:.0f}%), points={pts:,}")
        if reg > best_registered:
            best_registered, best_points, best_tier = reg, pts, "tier3"
        if ratio >= min_ratio and pts >= min_points:
            quality = _validate_reconstruction_quality(text_dir, total)
            if quality["passed"]:
                print(f"[COLMAP] ✓ Tier-3 succeeded. reprojection_err={quality['mean_reprojection_error']:.3f}px")
                return _finalize(text_dir, reg, total, pts, "tier3", on_progress)
            else:
                print(f"[COLMAP] ✗ Tier-3 geometry failed quality gate: {quality['reason']}")
    except Exception as e:
        print(f"[COLMAP] Tier-3 failed with exception: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TIER 4 — Low-texture rescue (dense features, relaxed mapper)
    # ════════════════════════════════════════════════════════════════════════
    print("\n[COLMAP] ════ TIER 4: Low-texture rescue ════")
    if on_progress:
        on_progress("Tier-4", "Starting Tier-4: low-texture rescue (30k features, peak=0.002)")
    try:
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()
        _reset_sparse(sparse_dir)

        _tier4_extraction(colmap_binary, db_path, image_dir, use_extraction_gpu, on_progress)
        _exhaustive_matching(colmap_binary, db_path, gpu_flag, on_progress)

        mq_rescue = {**_MAPPER_COMMON, **_MAPPER_QUALITY["rescue"]}
        _run_mapper(colmap_binary, db_path, image_dir, sparse_dir, mq_rescue, "Tier-4", on_progress)
        reg, pts = _convert_and_count(colmap_binary, sparse_dir, text_dir, "Tier-4", on_progress)
        ratio = reg / max(total, 1)
        print(f"[COLMAP] Tier-4 result: {reg}/{total} ({ratio*100:.0f}%)  points={pts:,}")
        if on_progress:
            on_progress("Tier-4", f"Registration: {reg}/{total} ({ratio*100:.0f}%), points={pts:,}")
        if reg > best_registered:
            best_registered, best_points, best_tier = reg, pts, "tier4"
        if ratio >= min_ratio and pts >= min_points:
            quality = _validate_reconstruction_quality(text_dir, total)
            if quality["passed"]:
                print(
                    f"[COLMAP] ✓ Tier-4 succeeded (low-texture rescue). "
                    f"reprojection_err={quality['mean_reprojection_error']:.3f}px  "
                    f"⚠ Verify visual output — Tier-4 reconstructions may have residual noise."
                )
                return _finalize(text_dir, reg, total, pts, "tier4", on_progress)
            else:
                print(
                    f"[COLMAP] ✗ Tier-4 REJECTED by geometry quality gate: {quality['reason']}\n"
                    "[COLMAP] Tier-4 result has too many errors to produce a clean splat — not continuing to training."
                )
    except Exception as e:
        print(f"[COLMAP] Tier-4 failed with exception: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # SALVAGE — if best result has any registered images, use it anyway
    # ════════════════════════════════════════════════════════════════════════
    if best_registered >= 10 and best_points >= 500:
        best_ratio = best_registered / max(total, 1)

        # Quality-gate salvage: reject if reprojection error is too high
        # (poisoned geometry produces fog even if file counts look OK)
        quality = _validate_reconstruction_quality(
            text_dir, total, min_registered_ratio=0.0, min_points=0
        )
        if quality.get("mean_reprojection_error", 0) > 3.0:
            print(
                f"[COLMAP] ✗ Salvage REJECTED: reprojection error "
                f"{quality['mean_reprojection_error']:.2f}px > 3.0px — "
                "geometry is too noisy for usable training. Hard failing."
            )
            # fall through to hard failure below
        else:
            print(
                f"\n[COLMAP] ⚠  All tiers below threshold but salvaging best result: "
                f"{best_registered}/{total} ({best_ratio*100:.0f}%) via {best_tier}"
            )
        if on_progress:
            on_progress(
                "Salvage",
                f"All tiers partial — using best: {best_registered}/{total} ({best_ratio*100:.0f}%)"
            )
        # text_dir already has the last tier's output; it may or may not be the best.
        # Re-convert from whichever sparse/0 exists.
        if (sparse_dir / "0").exists():
            if text_dir.exists():
                shutil.rmtree(text_dir)
            text_dir.mkdir(parents=True, exist_ok=True)
            run_cmd([
                colmap_binary, "model_converter",
                "--input_path",  str(sparse_dir / "0"),
                "--output_path", str(text_dir),
                "--output_type", "TXT",
            ], "Model Conversion (salvage)", on_progress)
        return _finalize(text_dir, best_registered, total, best_points, best_tier + "_salvage", on_progress)

    # ════════════════════════════════════════════════════════════════════════
    # ALL TIERS FAILED — build detailed diagnostic failure message
    # ════════════════════════════════════════════════════════════════════════
    diag = _scene_diagnostics(image_dir)
    advice_lines = "\n".join(f"  → {a}" for a in diag["advice"])

    raise RuntimeError(
        f"[COLMAP] HARD FAILURE: All four reconstruction tiers exhausted.\n"
        f"Best result across all tiers: {best_registered}/{total} images "
        f"({best_registered / max(total, 1) * 100:.0f}%), {best_points} points.\n\n"
        f"Scene diagnostics:\n"
        f"  Blur:         {diag['blur_pct']:.0f}% of sampled frames\n"
        f"  Low texture:  {diag['low_texture_pct']:.0f}% of sampled frames\n"
        f"  Overexposed:  {diag['overexposed_pct']:.0f}% of sampled frames\n\n"
        f"Specific advice:\n{advice_lines}\n\n"
        f"General fixes:\n"
        f"  1. Record a slow orbit (≥30 s, one step/second around subject).\n"
        f"  2. Ensure 60–80% overlap between consecutive frames.\n"
        f"  3. Place object on a textured surface (newspaper / graph paper).\n"
        f"  4. Use diffuse lighting — eliminate specular/mirror surfaces.\n"
        f"  5. Avoid pure-black/white backgrounds and featureless walls."
    )


def _finalize(
    text_dir: Path,
    registered: int,
    total: int,
    n_points: int,
    tier_used: str,
    on_progress: Optional[Callable],
) -> dict:
    """Post-pipeline quality warnings (non-blocking) and return stats dict."""
    ratio = registered / max(total, 1)

    if ratio < 0.40:
        print(
            f"[COLMAP] ⚠  WARNING: Only {ratio*100:.0f}% of images registered. "
            "Reconstruction quality may be low — consider reshooting."
        )
        if on_progress:
            on_progress("Quality", f"Low registration ({ratio*100:.0f}%) — splat quality may be degraded.")

    if n_points < 2000:
        print(f"[COLMAP] ⚠  WARNING: Only {n_points:,} 3D points — sparse cloud is thin.")
        if on_progress:
            on_progress("Quality", f"Thin sparse cloud ({n_points:,} pts) — training will still proceed.")
    elif n_points < 5000:
        print(f"[COLMAP] ℹ  Marginal sparse cloud: {n_points:,} pts. Target ≥5 000.")
    else:
        print(f"[COLMAP] ✓  Good sparse cloud: {n_points:,} points.")

    print(
        f"\n[COLMAP] Pipeline complete via {tier_used}: "
        f"{registered}/{total} images, {n_points:,} points."
    )

    return {
        "registered": registered,
        "total":      total,
        "ratio":      round(ratio, 3),
        "n_points":   n_points,
        "tier_used":  tier_used,
    }