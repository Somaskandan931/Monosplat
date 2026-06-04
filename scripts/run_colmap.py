"""
run_colmap.py
-------------
Run COLMAP sparse reconstruction on extracted frames.

Pipeline stages:
  1. feature_extractor  — SIFT feature extraction
  2. exhaustive_matcher — Pairwise feature matching
  3. mapper             — Sparse 3D reconstruction
  4. model_converter    — Convert to TXT format (cameras.txt, images.txt, points3D.txt)
  5. validate           — Check quality and warn if poor

Usage:
  python scripts/run_colmap.py --images data/frames --output data/sparse

"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ── COLMAP runner ─────────────────────────────────────────────────────────────

def run_command(cmd: list, step_name: str) -> bool:
    """Run a shell command, print output, return True if successful."""
    print(f"\n🔧 COLMAP: {step_name}")
    print(f"   Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ COLMAP {step_name} failed!")
        print("STDOUT:", result.stdout[-1000:])
        print("STDERR:", result.stderr[-1000:])
        return False

    print(f"✅ {step_name} done.")
    return True


def run_colmap(
    images_dir: str,
    output_dir: str,
    use_gpu: bool = True,
    camera_model: str = "SIMPLE_RADIAL",
) -> bool:
    """
    Run the full COLMAP sparse reconstruction pipeline.

    Args:
        images_dir:    Directory containing input JPEG frames.
        output_dir:    Directory to store sparse reconstruction output.
        use_gpu:       Use GPU for feature extraction/matching if available.
        camera_model:  COLMAP camera model. SIMPLE_RADIAL works well for monocular video.

    Returns:
        True if reconstruction succeeded and passes quality checks.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # COLMAP needs a database file and a sparse model folder
    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "0"

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"

    # ── Step 1: Feature Extraction (SIFT) ─────────────────────────────────────
    ok = run_command([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",    # Assume one camera (monocular)
        "--SiftExtraction.use_gpu", gpu_flag,
        # FIX: was 8192 → 16384.  More SIFT features per frame = more matches
        #      across frames = better chance of the mapper registering all frames.
        "--SiftExtraction.max_num_features", "16384",
    ], "Feature Extraction")

    if not ok:
        print("💡 Tip: Make sure COLMAP is installed: apt-get install -y colmap")
        return False

    # ── Step 2: Exhaustive Matching ────────────────────────────────────────────
    # Exhaustive matching checks ALL pairs — best for small datasets (<200 images)
    ok = run_command([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", gpu_flag,
        # FIX: raise max_num_matches (default 32768 is fine; but some builds
        #      default to 8192).  Explicit value ensures consistent behaviour
        #      regardless of COLMAP build defaults.
        "--SiftMatching.max_num_matches", "32768",
        # FIX: lower min_num_inliers so weakly-overlapping frame pairs survive
        #      into the match graph.  Default is 15; 8 helps near-boundary frames.
        "--TwoViewGeometry.min_num_inliers", "8",
    ], "Exhaustive Matching")

    if not ok:
        return False

    # ── Step 3: Sparse Reconstruction (Mapper) ─────────────────────────────────
    ok = run_command([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--output_path",   str(output_dir),
        "--Mapper.num_threads", "4",
        # FIX: was 4 → 1.5.  Lower init_min_tri_angle allows COLMAP to accept
        #      near-planar or forward-facing monocular video pairs that would
        #      otherwise all be rejected during the initial pair search.
        "--Mapper.init_min_tri_angle", "1.5",
        # FIX: removed multiple_models=0.  That flag forces COLMAP to keep the
        #      largest single reconstruction and discard all others.  For
        #      monocular video where the primary model repeatedly fails to grow
        #      past the minimum size, re-enabling multiple_models lets COLMAP
        #      accumulate fragment models and merge them.
        # "--Mapper.multiple_models", "0",  # REMOVED — see above
        # FIX: lower abs_pose_min_num_inliers so images that see only 10–20
        #      sparse points are not immediately rejected during registration.
        "--Mapper.abs_pose_min_num_inliers", "10",
        # FIX: lower min_num_matches to prevent pairs with few-but-good matches
        #      from being discarded in the match graph before the mapper sees them.
        "--Mapper.min_num_matches", "10",
        # FIX: cap global BA refinements to avoid runaway solver time on weak
        #      reconstructions; Eigen Cholesky failures (seen in logs) are a
        #      symptom of poorly-conditioned systems — fewer iterations helps.
        "--Mapper.ba_global_max_refinements", "3",
        # FIX: relax local BA inlier threshold for the same reason.
        "--Mapper.ba_local_max_refinements", "3",
    ], "Sparse Reconstruction (Mapper)")

    if not ok:
        print("\n💡 Reconstruction failed. Common causes:")
        print("   - Too few frames (need 20+)")
        print("   - Not enough overlap between frames")
        print("   - Too much camera motion between frames")
        print("   - Blurry or low-texture scene")
        return False

    # ── Step 4: Convert to TXT format ──────────────────────────────────────────
    # TXT format: cameras.txt, images.txt, points3D.txt — easy to parse in Python
    ok = run_command([
        "colmap", "model_converter",
        "--input_path",  str(sparse_dir),
        "--output_path", str(sparse_dir),
        "--output_type", "TXT",
    ], "Convert Model to TXT")

    if not ok:
        # Try to find another model folder (mapper sometimes creates 0, 1, 2...)
        alt_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if alt_dirs:
            alt = sorted(alt_dirs)[0]
            print(f"   Trying alternative model: {alt}")
            run_command([
                "colmap", "model_converter",
                "--input_path",  str(alt),
                "--output_path", str(sparse_dir),
                "--output_type", "TXT",
            ], "Convert Model to TXT (alt)")

    # ── Step 5: Scene Normalization ────────────────────────────────────────────
    # Normalizes camera positions to the unit ball so that training hyperparameters
    # (densify_grad_threshold, position_lr_init, etc.) work at the right scale.
    # Without this, large-scale scenes stretch geometry and cause splat explosion.
    _run_scene_normalization(sparse_dir)

    # ── Step 6: Quality Validation ─────────────────────────────────────────────
    return validate_reconstruction(images_dir, sparse_dir)


def _run_scene_normalization(sparse_dir: Path) -> None:
    """
    Compute and save camera-position normalization metadata.

    Reads images.txt from *sparse_dir*, computes the scene centre + scale,
    and writes scene_norm.json alongside the COLMAP text files.
    The normalisation is applied to camera centres only here; the trainer
    reads scene_norm.json and applies the same transform to the initial
    point cloud before training begins.
    """
    images_txt = sparse_dir / "images.txt"
    if not images_txt.exists():
        print("⚠️  Skipping scene normalization — images.txt not found.")
        return

    try:
        import sys
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[2]
        for _p in (str(_repo / "src"), str(_repo)):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from preprocessing.normalize_scene import (
            load_camera_positions,
            normalize_camera_positions,
            save_normalized_positions,
        )

        print("\n🔧 Scene Normalization")
        camera_positions = load_camera_positions(str(images_txt))

        normalized_positions, scene_center, scene_scale = normalize_camera_positions(
            camera_positions
        )

        norm_json = sparse_dir / "scene_norm.json"
        save_normalized_positions(
            normalized_positions, scene_center, scene_scale, str(norm_json)
        )
        print(f"✅ Scene normalization complete. scale={scene_scale:.4f}")

    except Exception as exc:
        print(f"⚠️  Scene normalization failed (non-fatal): {exc}")
        print("   Training will proceed without normalization.")


def validate_reconstruction(images_dir: Path, sparse_dir: Path) -> bool:
    """
    Validate that COLMAP produced a usable reconstruction.

    Checks:
    - images.txt exists and has registered cameras
    - points3D.txt exists and has enough points
    - Registration rate is above threshold
    """
    print("\n🔍 Validating COLMAP reconstruction...")

    images_txt  = sparse_dir / "images.txt"
    points3d_txt = sparse_dir / "points3D.txt"

    # Check files exist
    if not images_txt.exists():
        print("❌ images.txt not found. Reconstruction may have failed.")
        return False

    if not points3d_txt.exists():
        print("❌ points3D.txt not found.")
        return False

    # Count registered images
    registered = 0
    with open(images_txt) as f:
        for line in f:
            line = line.strip()
            # Image lines start with numeric ID (not #)
            if line and not line.startswith("#"):
                try:
                    int(line.split()[0])
                    registered += 1
                except ValueError:
                    pass
    # Each image takes 2 lines in images.txt
    registered = registered // 2

    # Count total images in the frames folder
    total_images = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    # Count sparse points
    sparse_points = 0
    with open(points3d_txt) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                sparse_points += 1

    # Compute registration rate
    reg_rate = registered / max(total_images, 1) * 100

    print(f"\n📊 Reconstruction Summary:")
    print(f"   Total images     : {total_images}")
    print(f"   Registered images: {registered} ({reg_rate:.1f}%)")
    print(f"   Sparse points    : {sparse_points:,}")

    # ── Thresholds ─────────────────────────────────────────────────────────────
    success = True

    if registered < 5:
        print("\n❌ CRITICAL: Only {registered} images registered.")
        print("   Training will likely fail. Please fix COLMAP first.")
        success = False

    elif reg_rate < 50:
        print(f"\n⚠️  WARNING: Low registration rate ({reg_rate:.1f}%).")
        print("   Less than half of your frames were registered.")
        print("   Consider: re-shooting with slower camera motion, more overlap.")
        # Still allow training, but warn

    if sparse_points < 100:
        print(f"\n⚠️  WARNING: Very few sparse points ({sparse_points}).")
        print("   The scene may be textureless or lighting is poor.")

    if success:
        print(f"\n✅ COLMAP reconstruction looks usable!")

    return success


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run COLMAP sparse reconstruction")
    parser.add_argument("--images",  required=True,            help="Frames directory")
    parser.add_argument("--output",  default="data/sparse",    help="Output sparse directory")
    parser.add_argument("--no_gpu",  action="store_true",      help="Disable GPU")
    parser.add_argument("--camera_model", default="SIMPLE_RADIAL", help="COLMAP camera model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_colmap(
        images_dir=args.images,
        output_dir=args.output,
        use_gpu=not args.no_gpu,
        camera_model=args.camera_model,
    )
    sys.exit(0 if success else 1)