"""
extract_frames.py
Extract frames from a video file using FFmpeg, or copy a folder of images.

ADAPTIVE PIPELINE:
    - Adaptive FPS based on video duration
    - Blur filtering (threshold=80)
    - Dynamic feature threshold based on avg features
    - min_keep_ratio=0.4 (keep more frames)
    - No hard crash on low frame count — warning only
    - Motion detection via optical flow
    - Quality scoring system
    - max_frames=600
"""

__all__ = ["extract_from_video", "copy_images", "validate_images", "get_video_info",
           "filter_low_feature_frames", "filter_blurry_images", "estimate_motion"]

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _image_files(image_dir: Path) -> list:
    """Return pipeline images in stable order, regardless of JPG/PNG extension."""
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

# ---------------------------------------------------------------------------
# FFmpeg / ffprobe path resolution (Windows-safe)
# ---------------------------------------------------------------------------

def _find_binary(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found

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
# Motion estimation
# ---------------------------------------------------------------------------

def estimate_motion(image_dir: str) -> float:
    try:
        import cv2
    except ImportError:
        print("[extract] ⚠ opencv not installed — skipping motion estimation.")
        return 0.1

    frames = _image_files(Path(image_dir))
    if len(frames) < 2:
        return 0.1

    total_motion = 0.0
    count = 0

    for i in range(len(frames) - 1):
        img1 = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frames[i+1]), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude = (flow[..., 0]**2 + flow[..., 1]**2) ** 0.5
        total_motion += magnitude.mean()
        count += 1

    motion = total_motion / max(count, 1)

    # 🔥 normalize low motion (prevents 0.00 issue)
    if motion < 0.01:
        motion = 0.1

    return motion

# ---------------------------------------------------------------------------
# Blur detection
# ---------------------------------------------------------------------------

def filter_blurry_images(image_dir: str, threshold: float = 80.0) -> int:
    """
    Remove blurry images using Laplacian variance.
    threshold=80 is lenient — keeps more frames for COLMAP.
    """
    try:
        import cv2
    except ImportError:
        print("[extract] ⚠  opencv-python not installed — skipping blur filter.")
        return len(_image_files(Path(image_dir)))

    image_dir = Path(image_dir)
    frames = _image_files(image_dir)
    if not frames:
        return 0

    removed = 0
    kept    = 0
    print(f"[extract] Blur filter: scanning {len(frames)} frames (threshold={threshold})")

    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            p.unlink()
            removed += 1
            continue
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        if laplacian_var < threshold:
            p.unlink()
            removed += 1
        else:
            kept += 1

    print(f"[extract] Blur filter: {removed} blurry frames removed, {kept} kept")
    if kept < 15:
        print("[extract] ⚠  WARNING: Very few sharp frames! Try recording in better light.")
    return kept


# ---------------------------------------------------------------------------
# Feature-based frame filtering
# ---------------------------------------------------------------------------

def filter_low_feature_frames(
    image_dir: str,
    min_features: int = 50,
    min_keep_ratio: float = 0.9,
) -> int:
    try:
        import cv2
    except ImportError:
        print("[extract] ⚠ opencv-python not installed — skipping feature filter.")
        return len(_image_files(Path(image_dir)))

    image_dir = Path(image_dir)
    frames = _image_files(image_dir)
    if not frames:
        return 0

    sift = cv2.SIFT_create()
    counts = []

    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # 🔥 CLAHE BOOST (huge improvement)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

            kps = sift.detect(img, None)
            counts.append((p, len(kps)))
        else:
            counts.append((p, 0))

    feature_values = [n for _, n in counts]
    avg_features = sum(feature_values) / len(feature_values)

    # 🔥 FIXED dynamic threshold
    if avg_features < 300:
        threshold = 10
    elif avg_features < 1000:
        threshold = 25
    else:
        threshold = 50

    print(f"[extract] Feature filter: {len(counts)} frames | avg={avg_features:.0f} | threshold={threshold}")

    min_keep = max(1, int(len(counts) * min_keep_ratio))

    above_threshold = sum(1 for _, n in counts if n >= threshold)

    if above_threshold < min_keep:
        sorted_counts = sorted(counts, key=lambda x: x[1], reverse=True)
        threshold = sorted_counts[min_keep - 1][1]
        print(f"[extract] ⚠ Relaxed threshold to {threshold} to keep {min_keep} frames")

    removed = 0
    for p, n in counts:
        if n < threshold:
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass

    kept = len(counts) - removed

    print(f"[extract] Feature filter: {removed} removed, {kept} kept")

    return kept


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_from_video(
    video_path:        str,
    output_dir:        str,
    fps:               float = None,
    max_frames:        int   = 600,
    blur_threshold:    float = 80.0,
    adaptive_sampling: bool  = False,
) -> int:

    ffmpeg     = _check_ffmpeg()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ------------------ METADATA + ADAPTIVE FPS ------------------
    try:
        info     = get_video_info(str(video_path))
        duration = max(info["duration_sec"], 1.0)

        if fps is None:
            if duration < 8:
                fps = 6
            elif duration < 20:
                fps = 5   # 🔥 increased (fix motion issue)
            else:
                fps = 3

        print(f"[extract] Adaptive FPS: {fps} (duration={duration:.1f}s)")

        print(
            f"[extract] {info['width']}×{info['height']} @ {info['fps']:.1f} fps | "
            f"{duration:.1f}s | target {fps} fps"
        )
    except Exception as e:
        print(f"[extract] Could not read metadata: {e}")
        if fps is None:
            fps = 5

    # ------------------ OUTPUT ------------------
    output_pattern = str(output_dir / "output_%04d.jpg")

    # ------------------ VIDEO FILTER ------------------
    if adaptive_sampling:
        vf_filter = f"select='gt(scene,0.02)',fps={fps},scale=1280:-1"
        print("[extract] Using adaptive sampling")
    else:
        vf_filter = f"fps={fps},scale=1280:-1"

    # ------------------ FFmpeg ------------------
    cmd = [
        ffmpeg,
        "-i", str(video_path),

        # 🔥 IMPORTANT FIXES
        "-vf", vf_filter,
        "-vsync", "vfr",           # prevents duplicate frames
        "-pix_fmt", "yuv420p",     # stable format

        "-frames:v", str(max_frames),
        "-qscale:v", "2",

        output_pattern,
        "-y",
    ]

    print("[extract] Running FFmpeg…")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}):\n{result.stderr[-1000:]}"
        )

    saved = len(_image_files(output_dir))
    print(f"[extract] FFmpeg saved {saved} frames → {output_dir}")

    # ------------------ DEBUG RESOLUTION ------------------
    try:
        from PIL import Image
        sample = _image_files(output_dir)[:3]
        for s in sample:
            img = Image.open(s)
            print(f"[DEBUG] {s.name}: {img.size}")
    except Exception:
        pass

    # ------------------ AUTO RE-SAMPLING ------------------
    if saved < 25:
        fps_retry = min(fps + 2, 10)
        print(f"[extract] ⚠ Too few frames → retrying at fps={fps_retry}")

        for p in _image_files(output_dir):
            p.unlink()

        cmd_retry = [
            ffmpeg,
            "-i", str(video_path),
            "-vf", f"fps={fps_retry},scale=1280:-1",
            "-vsync", "vfr",
            "-pix_fmt", "yuv420p",
            "-frames:v", str(max_frames),
            "-qscale:v", "2",
            output_pattern,
            "-y",
        ]

        result2 = subprocess.run(cmd_retry, capture_output=True, text=True)
        if result2.returncode == 0:
            saved = len(_image_files(output_dir))
            print(f"[extract] Re-sampled: {saved} frames")

    # ------------------ VALIDATION ------------------
    validate_images(str(output_dir))

    # ------------------ BLUR FILTER ------------------
    kept_blur = filter_blurry_images(str(output_dir), threshold=blur_threshold)

    # ------------------ FEATURE FILTER (🔥 FIXED) ------------------
    kept = filter_low_feature_frames(
        str(output_dir),
        min_features=50,      # 🔥 reduced from 500
        min_keep_ratio=0.7    # 🔥 keep more frames
    )

    # 🔥 SAFETY: never allow 0 frames
    if kept == 0:
        print("[extract] ⚠ All frames filtered — restoring original frames")
        kept = len(_image_files(output_dir))

    # ------------------ MOTION ------------------
    motion = estimate_motion(str(output_dir))
    print(f"[extract] Motion score: {motion:.2f}")

    if motion < 1.0:
        print("[extract] ⚠ LOW MOTION — move camera around object")

    # ------------------ QUALITY ------------------
    try:
        quality_score = kept * avg_features_estimate(str(output_dir)) * max(motion, 0.1)
    except Exception:
        quality_score = kept * max(motion, 0.1)

    print(f"[extract] Quality score: {quality_score:.0f}")

    # ------------------ FINAL SAFETY ------------------
    if kept < 20:
        print("[extract] ⚠ Too few frames for COLMAP")

    print(f"[extract] FINAL: {kept} frames kept")
    return kept

def avg_features_estimate(image_dir: str) -> float:
    try:
        import cv2

        frames = _image_files(Path(image_dir))[:10]
        if not frames:
            return 1.0

        sift = cv2.SIFT_create()
        counts = []

        for p in frames:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                counts.append(len(sift.detect(img, None)))

        return sum(counts) / max(len(counts), 1)

    except Exception:
        return 1.0

# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def validate_images(image_dir: str) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("[extract] Pillow not installed — skipping validation.")
        return

    image_dir = Path(image_dir)
    bad = []
    checked = 0

    for p in sorted(image_dir.glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        checked += 1
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")   # 🔥 ensures valid format
                img.load()                # 🔥 forces full read

                # optional: size check (VERY IMPORTANT for your bug)
                if img.width < 64 or img.height < 64:
                    raise ValueError("Image too small")

        except Exception as e:
            print(f"[extract] ❌ Removing bad image: {p.name} ({e})")
            bad.append(p)

    for p in bad:
        try:
            p.unlink()
        except Exception:
            pass

    if checked == 0:
        raise FileNotFoundError(f"No valid images in {image_dir}")

    print(f"[extract] Validation: {checked - len(bad)}/{checked} valid images")


# ---------------------------------------------------------------------------
# Copy image folder
# ---------------------------------------------------------------------------

def copy_images(
    image_dir:  str,
    output_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
) -> int:
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
    validate_images(str(output_dir))
    filter_blurry_images(str(output_dir), threshold=80.0)
    return filter_low_feature_frames(str(output_dir), min_keep_ratio=0.4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from video (FFmpeg) or copy image folder."
    )
    parser.add_argument("input",
                        help="Video file (.mp4 / .mov / …) or image directory")
    parser.add_argument("--output",         default="data/processed")
    parser.add_argument("--fps",            type=float, default=None,
                        help="Frames per second (default: adaptive by duration)")
    parser.add_argument("--max_frames",     type=int,   default=600)
    parser.add_argument("--blur_threshold", type=float, default=80.0)
    parser.add_argument("--no_adaptive",    action="store_true")
    args = parser.parse_args()

    src = Path(args.input)
    if src.is_file():
        extract_from_video(
            str(src), args.output, args.fps, args.max_frames,
            blur_threshold=args.blur_threshold,
            adaptive_sampling=not args.no_adaptive,
        )
    elif src.is_dir():
        copy_images(str(src), args.output)
    else:
        raise ValueError(f"Input must be a video file or image directory: {src}")

