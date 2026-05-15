"""
colmap_runner.py
Automates the COLMAP sparse-reconstruction pipeline.

KEY CHANGES vs previous version (fixing 46-Gaussian / empty-splat problem):
    1. exhaustive_matcher is now ALWAYS primary (not sequential_matcher).
       Sequential matching was the root cause — it ran in ~0.005s per frame,
       meaning it found almost zero real cross-view matches.
    2. SIFT peak_threshold lowered to 0.004 (default 0.0067) so weak-texture
       surfaces (dark logos, glossy products) produce more keypoints.
    3. max_num_features=16000 — dense enough for complex objects.
    4. SiftMatching.guided_matching=1 — epipolar constraint filters false matches,
       especially on reflective / repeated-pattern surfaces.
    5. SiftMatching.max_num_matches=32768 — allows rich matches per pair.
    6. Mapper thresholds stratified by quality (low / medium / high).
    7. 3D point count diagnostic: warns loudly if < 500 points found, which
       would produce an almost-empty Gaussian splat.

Output directory layout
-----------------------
    <output_dir>/
        database.db
        sparse/0/
        sparse_text/         <- cameras.txt, images.txt, points3D.txt
"""

import subprocess
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _detect_env(colmap_binary: str = "colmap") -> dict:
    from src.utils.env_detect import (
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

    Matching strategy (fixes the 46-Gaussian empty-splat problem):
        exhaustive_matcher — tries EVERY image pair.
        Sequential matching was the original default and was shown to take
        ~0.005 s per image, indicating effectively zero real feature matches
        were being produced. Exhaustive matching is the correct default for
        turntable / orbit captures where frame order does not imply
        visual proximity.

    SIFT changes:
        peak_threshold=0.004 (was default 0.0067) — finds weaker but real
        features on dark/low-texture objects like logos and glossy products.
        max_num_features=16000 — dense enough for complex objects.
        guided_matching=1 — epipolar constraint filters false matches.
        max_num_matches=32768 — allows rich matches between image pairs.
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

    env = _detect_env(colmap_binary)
    use_extraction_gpu = use_gpu and (force_gpu or env["use_extraction_gpu"])
    use_matching_gpu   = use_gpu and (force_gpu or env["use_matching_gpu"])

    print(
        f"[COLMAP] Auto-detect → "
        f"env={'colab' if env['in_colab'] else 'local'} "
        f"cuda_colmap={env['has_cuda_colmap']} "
        f"extraction_gpu={use_extraction_gpu} matching_gpu={use_matching_gpu}"
    )

    # Probe whether --SiftMatching.use_gpu is supported by this COLMAP build
    _matching_gpu_flag: list = []
    if use_matching_gpu:
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
                print("[COLMAP] Matching GPU flag not supported — using CPU matching")
        except Exception:
            pass

    # ---- 1. Feature extraction -------------------------------------------
    # peak_threshold=0.004 (default 0.0067): finds weaker features on dark /
    # low-texture objects. Crucial for logos and glossy products.
    extraction_cmd = [
        colmap_binary, "feature_extractor",
        "--database_path",                   db_path,
        "--image_path",                      str(image_dir),
        "--ImageReader.camera_model",        camera_model,
        "--ImageReader.single_camera",       "1" if single_camera else "0",
        "--SiftExtraction.max_num_features", "16000",
        "--SiftExtraction.peak_threshold",   "0.004",
        "--SiftExtraction.edge_threshold",   "10",
    ]
    if use_extraction_gpu:
        extraction_cmd += ["--SiftExtraction.use_gpu", "1"]
    _run_or_die(extraction_cmd, "Feature Extraction", on_progress)

    # ---- 2. Feature matching — exhaustive_matcher PRIMARY ----------------
    # exhaustive_matcher tries every image pair: O(N^2) but correct for
    # turntable / product orbit shoots where frame i+1 != visual neighbour.
    # guided_matching=1: uses epipolar geometry to filter out false matches.
    # max_num_matches=32768: allows dense matching for complex objects.
    exhaustive_cmd = [
        colmap_binary, "exhaustive_matcher",
        "--database_path",                db_path,
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.max_num_matches", "32768",
        "--SiftMatching.max_ratio",       "0.85",
    ] + _matching_gpu_flag
    rc = run_cmd(exhaustive_cmd, "Feature Matching (exhaustive)", on_progress)

    if rc != 0:
        # Fallback 1: sequential with high overlap — works when frame order ≈
        # visual proximity (continuous orbit captures, normal videos).
        print("[COLMAP] WARN  Exhaustive matching failed — trying sequential (overlap=40)")
        if on_progress:
            on_progress("Feature Matching", "Exhaustive failed — retrying sequential matcher (overlap=40)")
        sequential_cmd = [
            colmap_binary, "sequential_matcher",
            "--database_path",              db_path,
            "--SequentialMatching.overlap", "40",
        ] + _matching_gpu_flag
        rc2 = run_cmd(sequential_cmd, "Feature Matching (sequential fallback)", on_progress)

        if rc2 != 0:
            # Fallback 2: vocab_tree — appearance-based retrieval, best for
            # short/jumpy videos where neither exhaustive nor sequential work.
            # Requires a vocab tree binary; skip gracefully if absent.
            print("[COLMAP] WARN  Sequential matching also failed — trying vocab_tree_matcher")
            if on_progress:
                on_progress("Feature Matching", "Trying vocab-tree matcher as last resort")
            import shutil as _shutil
            vocab_tree_path = _shutil.which("vocab_tree_flickr100K_words32K.bin") or ""
            if not vocab_tree_path:
                # Try common install locations
                for _candidate in [
                    "/usr/share/colmap/vocab_tree_flickr100K_words32K.bin",
                    "/usr/local/share/colmap/vocab_tree_flickr100K_words32K.bin",
                ]:
                    if Path(_candidate).exists():
                        vocab_tree_path = _candidate
                        break
            if vocab_tree_path:
                vocab_cmd = [
                    colmap_binary, "vocab_tree_matcher",
                    "--database_path",                    db_path,
                    "--VocabTreeMatching.vocab_tree_path", vocab_tree_path,
                    "--VocabTreeMatching.num_images",     "30",
                ] + _matching_gpu_flag
                _run_or_die(vocab_cmd, "Feature Matching (vocab-tree fallback)", on_progress)
            else:
                print(
                    "[COLMAP] ⚠  vocab_tree binary not found — skipping. "
                    "Install with: apt-get install colmap-vocab-tree"
                )
                # Re-raise sequential failure so the caller knows
                raise RuntimeError(
                    "COLMAP feature matching failed with all three strategies "
                    "(exhaustive, sequential, vocab_tree). "
                    "The video likely has too little inter-frame overlap. "
                    "Record a slower, longer video (≥30 s, one step/second)."
                )

    # ---- 3. Sparse reconstruction (SfM) ----------------------------------
    # Thresholds stratified by quality level.
    # "low" is most permissive — used in auto-retry when registration < 50%.
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
            registered = 0
            data_line_idx = 0
            with open(images_txt) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # images.txt alternates: image-header line, keypoints line.
                    # Count only even-indexed data lines (the header lines).
                    if data_line_idx % 2 == 0:
                        registered += 1
                    data_line_idx += 1

            total = len(images)
            ratio = registered / max(total, 1)
            print(f"\n[COLMAP] Sparse Reconstruction: Registered {registered}/{total} images ({ratio*100:.0f}%)")

            if ratio < 0.5:
                print(
                    f"[COLMAP] ⚠  WARNING: only {registered}/{total} frames registered ({ratio*100:.0f}%). "
                    "Attempting low-quality auto-retry…"
                )
                if on_progress:
                    on_progress(
                        "Sparse Reconstruction",
                        f"WARNING: only {registered}/{total} frames registered ({ratio*100:.0f}%). Retrying with relaxed thresholds…"
                    )

                # Auto-retry with the most permissive mapper settings.
                # Clear the previous (incomplete) sparse model first.
                import shutil as _shutil
                for _d in sparse_dir.iterdir():
                    if _d.is_dir():
                        _shutil.rmtree(_d)
                    else:
                        _d.unlink()

                mq_low = _mapper_quality["low"]
                retry_mapper_cmd = [
                    colmap_binary, "mapper",
                    "--database_path",                       db_path,
                    "--image_path",                          str(image_dir),
                    "--output_path",                         str(sparse_dir),
                    "--Mapper.min_num_matches",              str(mq_low["min_num_matches"]),
                    "--Mapper.init_min_num_inliers",         str(mq_low["init_min_num_inliers"]),
                    "--Mapper.abs_pose_min_num_inliers",     str(mq_low["abs_pose_min_num_inliers"]),
                    "--Mapper.ba_global_function_tolerance", "0.000001",
                    "--Mapper.ba_local_max_num_iterations",  "15",
                    "--Mapper.ba_global_max_num_iterations", "30",
                    # Extra leniency for short / fast-motion videos
                    "--Mapper.min_focal_length_ratio",       "0.1",
                    "--Mapper.max_focal_length_ratio",       "10",
                    "--Mapper.max_extra_param",              "1",
                ]
                rc_retry = run_cmd(retry_mapper_cmd, "Sparse Reconstruction (low-quality retry)", on_progress)
                if rc_retry == 0:
                    # Re-run model conversion on the new output
                    model_src_retry = sparse_dir / "0"
                    if model_src_retry.exists():
                        import shutil as _sh2
                        if text_dir.exists():
                            _sh2.rmtree(text_dir)
                        text_dir.mkdir(parents=True, exist_ok=True)
                        run_cmd([
                            colmap_binary, "model_converter",
                            "--input_path",  str(model_src_retry),
                            "--output_path", str(text_dir),
                            "--output_type", "TXT",
                        ], "Model Conversion (retry)", on_progress)
                        print("[COLMAP] ✓  Low-quality retry succeeded — check registration stats below.")
                    else:
                        print("[COLMAP] ⚠  Low-quality retry produced no model at sparse/0.")
                else:
                    print(
                        "[COLMAP] ⚠  Low-quality retry also failed. "
                        "Record a slower video (≥30 s, one step/second around the subject)."
                    )
    except Exception as e:
        print(f"[COLMAP] Could not parse registration stats: {e}")

    # ---- 6. 3D point count diagnostic ------------------------------------
    # This is the most direct indicator of whether training will produce a
    # usable Gaussian splat. < 500 points = almost certainly empty render.
    try:
        points3d_txt = text_dir / "points3D.txt"
        if points3d_txt.exists():
            n_points = sum(
                1 for line in open(points3d_txt)
                if line.strip() and not line.startswith("#")
            )
            print(f"[COLMAP] 3D points in sparse model: {n_points:,}")

            if n_points < 500:
                msg = (
                    f"[COLMAP] ⚠  CRITICAL: Only {n_points} 3D points found.\n"
                    "  This will produce an almost-empty Gaussian splat (like the 46-splat case).\n"
                    "  Fix:\n"
                    "    1. Reshoot with 40-80 images or a slow-orbit video.\n"
                    "    2. Use diffuse lighting — eliminate specular highlights.\n"
                    "    3. Place object on a textured surface (newspaper, graph paper).\n"
                    "    4. Ensure 60-80% overlap between consecutive frames.\n"
                    "    5. Avoid pure-black backgrounds."
                )
                print(msg)
                if on_progress:
                    on_progress(
                        "Sparse Reconstruction",
                        f"CRITICAL: Only {n_points} 3D points — splat will be nearly empty. Reshoot with better lighting and 60%+ overlap."
                    )
            elif n_points < 5000:
                print(f"[COLMAP] ⚠  MARGINAL: {n_points} points — splat quality may be low.")
            else:
                print(f"[COLMAP] ✓  Good sparse cloud: {n_points:,} points.")
    except Exception as e:
        print(f"[COLMAP] Could not count 3D points: {e}")

    print(f"[COLMAP] COLMAP Complete ✓")