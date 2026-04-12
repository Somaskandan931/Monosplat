"""
extract_frames.py
Extract frames from a video file using FFmpeg, or copy a folder of images.

Output goes to:  data/processed/   (configured via --output CLI arg or caller)

FFmpeg is used instead of OpenCV for:
  - Broader codec support (H.264, H.265, ProRes, etc.)
  - Faster extraction (hardware-accelerated on some platforms)
  - No OpenCV dependency

Windows note: FFmpeg path resolution checks common install locations if the
              binary is not on the system PATH.

Public API
----------
    extract_from_video   — extract frames from a video file
    copy_images          — copy a folder of images
    validate_images      — validate and remove corrupted frames
    get_video_info       — return video metadata dict
"""

__all__ = ["extract_from_video", "copy_images", "validate_images", "get_video_info",
           "filter_low_feature_frames", "filter_blurry_images"]

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# FFmpeg / ffprobe path resolution (Windows-safe)
# ---------------------------------------------------------------------------

def _find_binary(name: str) -> str:
    """
    Locate *name* (ffmpeg or ffprobe) on PATH or common Windows locations.

    Returns:
        Full path string to the executable.

    Raises:
        RuntimeError: If the binary cannot be found.
    """
    # 1. Check PATH first (works on Linux, macOS, and correctly set-up Windows)
    found = shutil.which(name)
    if found:
        return found

    # 2. Common Windows install locations
    exe = name + ".exe"
    candidates = [
        rf"C:\ffmpeg\bin\{exe}",
        rf"C:\Program Files\ffmpeg\bin\{exe}",
        rf"C:\Program Files (x86)\ffmpeg\bin\{exe}",
        os.path.join(os.environ.get("USERPROFILE",    ""), rf"ffmpeg\bin\{exe}"),
        os.path.join(os.environ.get("LOCALAPPDATA",   ""), rf"ffmpeg\bin\{exe}"),
        os.path.join(os.environ.get("ProgramFiles",   ""), rf"ffmpeg\bin\{exe}"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), rf"ffmpeg\bin\{exe}"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path

    # 3. For ffprobe: derive from ffmpeg path if found
    if name == "ffprobe":
        try:
            ffmpeg_path = _find_binary("ffmpeg")
            ffprobe_path = str(Path(ffmpeg_path).parent / exe)
            if Path(ffprobe_path).exists():
                return ffprobe_path
        except RuntimeError:
            pass

    raise RuntimeError(
        f"{name} not found on PATH or common Windows install locations.\n"
        "Install it:\n"
        "  Linux:   sudo apt install ffmpeg\n"
        "  macOS:   brew install ffmpeg\n"
        "  Windows: https://ffmpeg.org/download.html  →  extract to C:\\ffmpeg\\"
    )


def _check_ffmpeg() -> str:
    """Verify FFmpeg is available; return its path."""
    path   = _find_binary("ffmpeg")
    result = subprocess.run([path, "-version"], capture_output=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg found at {path} but failed to execute.")
    print(f"[extract] FFmpeg: {path}")
    return path


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    """
    Return basic video metadata using ffprobe.

    Returns:
        dict with keys: duration_sec, fps, width, height, total_frames
    """
    ffprobe = _find_binary("ffprobe")
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}:\n{result.stderr[:500]}")

    data         = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        raise RuntimeError(f"No video stream found in {video_path}")

    fps_raw  = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_raw.split("/")
    fps      = float(num) / float(den)
    duration = float(video_stream.get("duration", 0))

    return {
        "duration_sec": duration,
        "fps":          fps,
        "width":        int(video_stream.get("width",  0)),
        "height":       int(video_stream.get("height", 0)),
        "total_frames": int(duration * fps),
    }


# ---------------------------------------------------------------------------
# Blur detection (NEW)
# ---------------------------------------------------------------------------

def filter_blurry_images(image_dir: str, threshold: float = 120.0) -> int:
    """
    Remove blurry images using Laplacian variance method.

    Args:
        image_dir: Directory containing extracted frames
        threshold: Minimum Laplacian variance to keep image (higher = sharper)
                   Default 120 works well; reduce to 80 for low-light scenes

    Returns:
        Number of images kept

    Requires: opencv-python (cv2)
    """
    try:
        import cv2
    except ImportError:
        print("[extract] ⚠  opencv-python not installed — skipping blur filter.\n"
              "           Install with: pip install opencv-python-headless")
        return len(list(Path(image_dir).glob("*.png")))

    image_dir = Path(image_dir)
    frames = sorted(image_dir.glob("*.png"))
    if not frames:
        return 0

    removed = 0
    kept = 0

    print(f"[extract] Blur filter: scanning {len(frames)} frames (threshold={threshold})")

    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            p.unlink()
            removed += 1
            continue

        # Laplacian variance measures image sharpness
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

        if laplacian_var < threshold:
            p.unlink()
            removed += 1
        else:
            kept += 1

    print(f"[extract] Blur filter: {removed} blurry frames removed, {kept} kept")

    if kept < 10:
        print("[extract] ⚠  WARNING: Very few sharp frames! Try reducing threshold to 80")

    return kept


# ---------------------------------------------------------------------------
# Frame extraction with smart sampling
# ---------------------------------------------------------------------------

def extract_from_video(
    video_path:  str,
    output_dir:  str,
    fps:         float = 3.0,
    max_frames:  int   = 400,
    blur_threshold: float = 50.0,   # very lenient — COLMAP handles blurry better than missing frames
    adaptive_sampling: bool = False, # disabled — steady fps gives better COLMAP overlap
) -> int:
    """
    Extract frames from *video_path* using FFmpeg with smart sampling.

    Output files are named  output_0001.png, output_0002.png, …
    and are written to *output_dir* (created if needed).

    IMPROVEMENTS:
        - Lower default FPS (2) for cleaner COLMAP matching
        - Optional scene-change detection (adaptive_sampling)
        - Automatic blur filtering after extraction

    Args:
        video_path: Input video (.mp4 / .mov / .avi / etc.).
        output_dir: Destination directory  →  data/processed/
        fps:        Frames per second to extract (default 2 — optimal for COLMAP).
        max_frames: Hard cap on total frames extracted (default 200).
        blur_threshold: Minimum sharpness to keep frame (default 120).
        adaptive_sampling: Use scene-change detection (removes redundant frames).

    Returns:
        Number of PNG frames saved (after blur filtering).
    """
    ffmpeg     = _check_ffmpeg()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Log video metadata (best-effort)
    try:
        info = get_video_info(str(video_path))
        print(
            f"[extract] {info['width']}×{info['height']} @ {info['fps']:.1f} fps  "
            f"| {info['duration_sec']:.1f}s  "
            f"| target {fps} fps  "
            f"| ~{int(info['duration_sec'] * fps)} frames"
        )
    except Exception as e:
        print(f"[extract] Could not read metadata: {e}")

    output_pattern = str(output_dir / "output_%04d.png")

    # Build FFmpeg filter chain
    if adaptive_sampling:
        # Scene-change detection + FPS sampling
        vf_filter = f"select='gt(scene,0.02)',fps={fps}"
        print("[extract] Using adaptive sampling (scene-change detection)")
    else:
        vf_filter = f"fps={fps}"

    cmd = [
        ffmpeg,
        "-i",        str(video_path),
        "-vf",       vf_filter,
        "-q:v",      "2",            # high quality PNG compression
        "-frames:v", str(max_frames),
        output_pattern,
        "-y",                        # overwrite existing frames
    ]

    print("[extract] Running FFmpeg…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}):\n{result.stderr[-1000:]}"
        )

    saved = len(list(output_dir.glob("output_*.png")))
    print(f"[extract] FFmpeg saved {saved} frames → {output_dir}")

    # Validate images (remove corrupted)
    validate_images(str(output_dir))

    # NEW: Remove blurry frames
    kept = filter_blurry_images(str(output_dir), threshold=blur_threshold)

    # NEW: Remove low-feature frames (if OpenCV available)
    kept = filter_low_feature_frames(str(output_dir), min_features=3000, min_keep_ratio=0.5)

    print(f"[extract] FINAL: {kept} high-quality frames kept")
    return kept


def validate_images(image_dir: str) -> None:
    """
    Fix 4 — Validate images before passing to COLMAP.
    Checks that every image in *image_dir* is a readable, non-corrupted
    PNG or JPEG.  Removes any unreadable file and prints a warning.
    Requires Pillow.
    """
    try:
        from PIL import Image as _PILImage
    except ImportError:
        print("[extract] Pillow not installed — skipping image validation.")
        return

    image_dir = Path(image_dir)
    bad = []
    checked = 0
    for p in sorted(image_dir.glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        checked += 1
        try:
            with _PILImage.open(p) as img:
                img.verify()          # detects truncated / corrupted files
        except Exception as e:
            print(f"[extract] ⚠  Corrupted / unreadable image removed: {p.name}  ({e})")
            bad.append(p)

    for p in bad:
        p.unlink()

    if checked == 0:
        raise FileNotFoundError(f"No .png / .jpg images found in {image_dir}")

    print(
        f"[extract] Image validation: {checked} images scanned, "
        f"{len(bad)} corrupted removed, {checked - len(bad)} valid."
    )


def filter_low_feature_frames(
    image_dir: str,
    min_features: int = 500,        # lowered from 3000 — keep more frames for COLMAP
    min_keep_ratio: float = 0.7,    # keep at least 70% of frames
) -> int:
    """
    Remove frames that are too featureless for COLMAP to use.

    Uses OpenCV's SIFT detector to count keypoints per frame, then removes
    any frame below *min_features*. This prevents the mapper from wasting
    time on blank/blurry/panned-away frames and avoids the Cholesky
    factorization failures seen when bad frames dilute the point cloud.

    A safety floor is enforced: if more than (1 - min_keep_ratio) of frames
    would be removed, the threshold is relaxed so at least min_keep_ratio of
    the original frames survive. This prevents accidentally deleting everything
    if the entire video is low-texture.

    Args:
        image_dir:      Directory of extracted PNG frames.
        min_features:   Minimum SIFT keypoints required to keep a frame.
                        Default 3000 — well above the ~500 seen on blank frames,
                        well below the ~8000+ seen on good object frames.
        min_keep_ratio: Minimum fraction of frames to retain (default 0.5).
                        Ensures at least half the frames survive even on
                        low-texture scenes.

    Returns:
        Number of frames kept.

    Requires: opencv-python  (cv2)
    """
    try:
        import cv2
    except ImportError:
        print(
            "[extract] ⚠  opencv-python not installed — skipping feature filter.\n"
            "           Install with: pip install opencv-python-headless"
        )
        return len(list(Path(image_dir).glob("output_*.png")))

    image_dir = Path(image_dir)
    frames = sorted(image_dir.glob("output_*.png"))
    if not frames:
        return 0

    sift = cv2.SIFT_create()

    counts: list[tuple[Path, int]] = []
    for p in frames:
        img  = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            counts.append((p, 0))
            continue
        kps = sift.detect(img, None)
        counts.append((p, len(kps)))

    counts.sort(key=lambda x: x[1])

    # Log the feature distribution so users can see the cliff in the console
    print(f"[extract] Feature filter: {len(counts)} frames scanned")
    low  = sum(1 for _, n in counts if n < min_features)
    high = len(counts) - low
    print(f"[extract]   ≥{min_features} features: {high} frames  |  <{min_features}: {low} frames")

    # Safety floor: never remove more than (1 - min_keep_ratio) of all frames
    min_keep  = max(1, int(len(counts) * min_keep_ratio))
    threshold = min_features

    if high < min_keep:
        # Too many frames below threshold — relax threshold to keep min_keep frames
        # (take the min_keep-th highest count as the new threshold)
        threshold = counts[-(min_keep)][1]
        print(
            f"[extract] ⚠  Relaxing threshold to {threshold} features "
            f"to keep at least {min_keep} frames (min_keep_ratio={min_keep_ratio})"
        )

    removed = 0
    for p, n in counts:
        if n < threshold:
            p.unlink()
            removed += 1

    kept = len(counts) - removed
    print(
        f"[extract] Feature filter done: {removed} low-feature frames removed, "
        f"{kept} frames kept."
    )

    if kept < 10:
        raise RuntimeError(
            f"[extract] Only {kept} frames remain after feature filtering — "
            "too few for COLMAP. Record the object more carefully: keep the "
            "camera focused on the object for the entire video."
        )

    return kept


def copy_images(
    image_dir:  str,
    output_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
) -> int:
    """
    Copy images from *image_dir* into *output_dir* with sequential naming.

    Useful when the user supplies a folder of photos instead of a video.

    Args:
        image_dir:  Source folder containing images.
        output_dir: Destination  →  data/processed/
        extensions: Accepted file extensions.

    Returns:
        Number of images copied.
    """
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.suffix in extensions)
    if not images:
        raise FileNotFoundError(f"No images with extensions {extensions} in {image_dir}")

    for i, src in enumerate(images):
        dst = output_dir / f"output_{i + 1:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        if i % 50 == 0:
            print(f"[extract] Copying {i + 1}/{len(images)}…")

    print(f"[extract] Copied {len(images)} images → {output_dir}")

    # Still apply blur and feature filtering to images
    validate_images(str(output_dir))
    filter_blurry_images(str(output_dir), threshold=120)
    return filter_low_feature_frames(str(output_dir))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from video (FFmpeg) or copy image folder."
    )
    parser.add_argument("input",
                        help="Video file (.mp4 / .mov / …) or image directory")
    parser.add_argument("--output",     default="data/processed",
                        help="Output directory for frames (default: data/processed)")
    parser.add_argument("--fps",        type=float, default=2.0,
                        help="Frames per second to extract (default: 2 — optimal for COLMAP)")
    parser.add_argument("--max_frames", type=int,   default=200,
                        help="Maximum frames to extract (default: 200)")
    parser.add_argument("--blur_threshold", type=float, default=120.0,
                        help="Blur detection threshold (higher = sharper, default: 120)")
    parser.add_argument("--no_adaptive", action="store_true",
                        help="Disable scene-change adaptive sampling")
    args = parser.parse_args()

    src = Path(args.input)
    if src.is_file():
        extract_from_video(
            str(src), args.output, args.fps, args.max_frames,
            blur_threshold=args.blur_threshold,
            adaptive_sampling=not args.no_adaptive
        )
    elif src.is_dir():
        copy_images(str(src), args.output)
    else:
        raise ValueError(f"Input must be a video file or image directory: {src}")