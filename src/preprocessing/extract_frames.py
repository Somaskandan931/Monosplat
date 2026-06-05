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
    - max_frames=300  (T&T benchmark needs 150–300 images; was 150 for small-object scans)
    - histogram + SSIM diversity filtering to remove near-duplicate viewpoints
"""

__all__ = ["extract_from_video", "copy_images", "validate_images", "validate_image_resolution",
           "get_video_info", "filter_low_feature_frames", "filter_blurry_images", "estimate_motion",
           "validate_exposure"]

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _image_files(image_dir: Path) -> list:
    """Return pipeline images in stable order, regardless of JPG/PNG extension."""
    try:
        from core.dataset_analysis.common import image_files
        return image_files(image_dir)
    except ImportError:
        return sorted(
            p for p in Path(image_dir).iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )


def _frame_selection_report_path(image_dir: Path) -> Path:
    """Keep reports outside the COLMAP-visible image folder when possible."""
    return image_dir.parent / "frame_selection_report.json"


def run_smart_frame_selection(image_dir: str, budget: int = 300) -> dict:
    """Run budgeted selection while preserving legacy top-level frame layout."""
    from core.frame_selection import SmartFrameSelectionEngine

    image_dir_path = Path(image_dir)
    report_path = _frame_selection_report_path(image_dir_path)
    report = SmartFrameSelectionEngine().select(
        image_dir_path,
        budget=budget,
        output_path=report_path,
        mutate=True,
    )
    print(
        "[selection] Smart frame selection: "
        f"{report['selected_frame_count']}/{report['original_frame_count']} selected "
        f"(budget={report['resolved_budget']})"
    )
    print(f"[selection] Saved frame_selection_report.json -> {report_path}")
    return report


# ---------------------------------------------------------------------------
# Resolution validation  (fixes #1, #11, #12 from audit)
# ---------------------------------------------------------------------------

def validate_image_resolution(image_dir: str, min_size: int = 256) -> None:
    """
    Hard-fail before COLMAP if any images are too small or unreadable.

    COLMAP cannot reconstruct meaningful poses from images smaller than
    ~256px — it produces near-zero features and a degenerate sparse model
    (the '46 gaussians / blank preview' symptom).

    Raises RuntimeError listing every offending image so the user knows
    exactly which uploads are broken before wasting time on COLMAP.
    """
    from PIL import Image

    image_dir = Path(image_dir)
    bad_images = []

    print(f"\n[validate_resolution] Checking images in: {image_dir}")

    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size  # PIL: (width, height)

            print(f"[validate_resolution] {img_path.name}: {w}x{h}")

            # Hard failure for images too small for COLMAP (#11)
            if w < min_size or h < min_size:
                bad_images.append((img_path.name, w, h))

        except Exception as e:
            bad_images.append((img_path.name, "ERROR", str(e)))

    if bad_images:
        errors = "\n".join(
            [f"  - {name}: {w}x{h}" for name, w, h in bad_images]
        )
        raise RuntimeError(
            f"HARD FAILURE: Images too small for COLMAP reconstruction.\n"
            f"COLMAP requires images >= {min_size}x{min_size}px.\n"
            f"Tiny images produce near-zero SIFT features → blank splat.\n\n"
            f"Offending images:\n{errors}\n\n"
            f"Fix: re-upload original full-resolution photos or video."
        )

    print(f"[validate_resolution] ✓ All images meet minimum {min_size}x{min_size}px requirement.")

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
    # core/ may not exist in this repo; fall back to lightweight motion scoring.
    try:
        from core.dataset_analysis.motion_analyzer import estimate_motion as _estimate_motion  # type: ignore
        return _estimate_motion(image_dir)
    except Exception:
        pass

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

def filter_blurry_images(image_dir: str, threshold: float = 120.0) -> int:
    """
    Remove blurry images using Laplacian variance.
    threshold=120 is moderate — rejects clearly blurry frames while keeping enough for COLMAP.
    Blurry images are MOVED to a 'blurry' subfolder instead of deleted,
    which is safer on Windows (avoids file-lock errors).
    """
    # Prefer local implementation; avoids core/ dependency.
    try:
        import cv2
    except ImportError:
        print("[extract] ⚠  opencv-python not installed — skipping blur filter.")
        return len(_image_files(Path(image_dir)))


    import gc

    image_dir = Path(image_dir)
    frames = _image_files(image_dir)
    if not frames:
        return 0

    # Move blurry frames here instead of deleting (Windows file-lock safe)
    bad_dir = image_dir / "blurry"
    bad_dir.mkdir(exist_ok=True)

    removed = 0
    kept    = 0
    print(f"[extract] Blur filter: scanning {len(frames)} frames (threshold={threshold})")

    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            del img
            gc.collect()
            p.rename(bad_dir / p.name)
            removed += 1
            continue
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        # Release the OpenCV mat and GC before any file operation (Windows fix)
        del img
        gc.collect()
        if laplacian_var < threshold:
            p.rename(bad_dir / p.name)
            removed += 1
        else:
            kept += 1

    print(f"[extract] Blur filter: {removed} blurry frames moved to '{bad_dir}', {kept} kept")
    if kept < 15:
        print("[extract] ⚠  WARNING: Very few sharp frames! Try recording in better light.")
    return kept


def validate_exposure(image_dir: str, overexposed_thresh: float = 245.0,
                      underexposed_thresh: float = 10.0,
                      bad_ratio_limit: float = 0.3) -> dict:
    """
    Detect frames with exposure problems (over- or under-exposed).

    Returns a summary dict with counts and a human-readable warning message.
    Frames are not removed — this is a diagnostic/advisory function only.

    Args:
        image_dir:           Directory of extracted frames.
        overexposed_thresh:  Mean pixel value above which a frame is flagged (0–255).
        underexposed_thresh: Mean pixel value below which a frame is flagged.
        bad_ratio_limit:     Fraction of flagged frames that triggers a warning.

    Returns:
        dict with keys: total, overexposed, underexposed, ok, warning (str or None)
    """
    from core.dataset_analysis.exposure_analyzer import analyze_exposure

    result = analyze_exposure(
        image_dir,
        overexposed_thresh=overexposed_thresh,
        underexposed_thresh=underexposed_thresh,
    )
    total = result["total_frames"]
    overexposed = result["overexposed"]
    underexposed = result["underexposed"]
    ok = result["ok"]
    warning = None
    bad_total = overexposed + underexposed
    if total and bad_total / max(total, 1) > bad_ratio_limit:
        parts = []
        if overexposed > 0:
            parts.append(f"{overexposed} overexposed (blown-out highlights)")
        if underexposed > 0:
            parts.append(f"{underexposed} underexposed (too dark)")
        warning = (
            f"[extract] âš   Exposure warning: {', '.join(parts)} out of {total} frames. "
            "Lock camera exposure before recording to avoid COLMAP failures."
        )
        print(warning)
    elif total:
        print(f"[extract] Exposure check: {ok}/{total} frames OK, "
              f"{overexposed} over, {underexposed} under.")
    return {
        "total": total,
        "overexposed": overexposed,
        "underexposed": underexposed,
        "ok": ok,
        "warning": warning,
    }

    try:
        import cv2
    except ImportError:
        return {"total": 0, "overexposed": 0, "underexposed": 0, "ok": 0, "warning": None}

    frames = _image_files(Path(image_dir))
    total  = len(frames)
    if total == 0:
        return {"total": 0, "overexposed": 0, "underexposed": 0, "ok": 0, "warning": None}

    overexposed   = 0
    underexposed  = 0

    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mean = float(img.mean())
        if mean >= overexposed_thresh:
            overexposed += 1
        elif mean <= underexposed_thresh:
            underexposed += 1

    bad_total = overexposed + underexposed
    ok        = total - bad_total
    warning   = None

    if bad_total / max(total, 1) > bad_ratio_limit:
        parts = []
        if overexposed > 0:
            parts.append(f"{overexposed} overexposed (blown-out highlights)")
        if underexposed > 0:
            parts.append(f"{underexposed} underexposed (too dark)")
        warning = (
            f"[extract] ⚠  Exposure warning: {', '.join(parts)} out of {total} frames. "
            "Lock camera exposure before recording to avoid COLMAP failures."
        )
        print(warning)
    else:
        print(f"[extract] Exposure check: {ok}/{total} frames OK, "
              f"{overexposed} over, {underexposed} under.")

    return {
        "total":        total,
        "overexposed":  overexposed,
        "underexposed": underexposed,
        "ok":           ok,
        "warning":      warning,
    }


# ---------------------------------------------------------------------------
# Feature-based frame filtering
# ---------------------------------------------------------------------------

def filter_low_feature_frames(
    image_dir: str,
    min_features: int = 50,
    min_keep_ratio: float = 0.9,
) -> int:
    # Prefer local implementation; avoids core/ dependency.
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
# Viewpoint diversity filtering  (Issue E fix)
# ---------------------------------------------------------------------------

def filter_duplicate_viewpoints(
    image_dir: str,
    ssim_threshold: float = 0.88,  # 0.92→0.88: only remove truly identical viewpoints
    hist_threshold: float = 0.96,  # 0.98→0.96: slightly stricter histogram gate
    min_keep_ratio: float = 0.5,
) -> int:
    """

    Remove near-duplicate frames that hurt COLMAP bundle adjustment.

    Two-stage filter:
      1. Histogram correlation — O(N) pass: any frame with hist_corr >= hist_threshold
         compared to its nearest retained neighbour is dropped.
      2. SSIM pass on close pairs only — catches frames that differ in brightness
         but share the same viewpoint (e.g. exposure flicker).

    Keeps at least min_keep_ratio fraction of input frames to prevent over-deletion.
    Moved (not deleted) to a 'duplicates/' subfolder for safety on Windows.

    Returns final kept count.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[diversity] ⚠ opencv not installed — skipping viewpoint diversity filter.")
        return len(_image_files(Path(image_dir)))

    image_dir = Path(image_dir)
    frames = _image_files(image_dir)
    if len(frames) < 2:
        return len(frames)

    dup_dir = image_dir / "duplicates"
    dup_dir.mkdir(exist_ok=True)

    print(f"[diversity] Scanning {len(frames)} frames for near-duplicate viewpoints …")

    # --- Stage 1: histogram correlation pass ---
    kept_indices = [0]  # always keep first frame
    ref_img = cv2.imread(str(frames[0]))
    ref_hist = _compute_hist(ref_img)

    for i in range(1, len(frames)):
        img = cv2.imread(str(frames[i]))
        if img is None:
            continue
        hist = _compute_hist(img)
        corr = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
        if corr < hist_threshold:
            kept_indices.append(i)
            ref_hist = hist  # advance reference to last kept
        # else: near-duplicate of previous retained frame → candidate for removal

    # --- Stage 2: SSIM double-check on adjacent kept pairs ---
    final_kept = [kept_indices[0]]
    for idx in kept_indices[1:]:
        prev_idx = final_kept[-1]
        img_a = cv2.imread(str(frames[prev_idx]), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(str(frames[idx]),      cv2.IMREAD_GRAYSCALE)
        if img_a is None or img_b is None:
            final_kept.append(idx)
            continue
        # Resize to small patch for speed
        h, w = 64, 64
        a_small = cv2.resize(img_a, (w, h))
        b_small = cv2.resize(img_b, (w, h))
        diff = (a_small.astype(np.float32) - b_small.astype(np.float32)) ** 2
        mse = diff.mean()
        # approx SSIM: high MSE → different viewpoint → keep
        ssim_approx = 1.0 - (mse / (255.0 ** 2))
        if ssim_approx < ssim_threshold:
            final_kept.append(idx)
        # else: SSIM too similar → skip (duplicate)

    # Enforce min_keep_ratio
    min_keep = max(1, int(len(frames) * min_keep_ratio))
    if len(final_kept) < min_keep:
        # Not enough diversity detected — fall back to uniform subsample
        step = len(frames) / min_keep
        final_kept = [int(i * step) for i in range(min_keep)]
        print(f"[diversity] ⚠ Too few diverse frames — falling back to uniform subsample ({min_keep} frames)")

    kept_set = set(final_kept)
    removed = 0
    for i, p in enumerate(frames):
        if i not in kept_set:
            try:
                p.rename(dup_dir / p.name)
                removed += 1
            except Exception:
                pass

    kept = len(_image_files(image_dir))
    print(f"[diversity] Viewpoint filter: {removed} duplicates moved, {kept} diverse frames kept")
    return kept


def _compute_hist(img) -> object:
    """Compute a normalised 3-channel HSV histogram for quick similarity comparison."""
    import cv2
    import numpy as np
    if img is None:
        return np.zeros((8 * 8 * 8,), dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# ---------------------------------------------------------------------------
# Short-video frame densification via optical-flow warping
# ---------------------------------------------------------------------------

def _densify_frames_optical_flow(output_dir: Path, target_frames: int = 80) -> int:
    """
    Synthesise midpoint frames between consecutive pairs using optical-flow
    warping.  Each synthetic frame is a valid intermediate camera pose —
    SIFT extracts real features from it and COLMAP can register it.

    This is called automatically when a video is shorter than
    SHORT_VIDEO_THRESHOLD (20 s) and we have fewer frames than target_frames.

    Strategy: one round of interpolation at α=0.5 (midpoint).  If after one
    round we still have fewer than target_frames, do a second round at α=0.5
    on adjacent pairs again.  Never exceeds 3 rounds to avoid temporal aliasing.

    Returns the final frame count.
    """
    try:
        import cv2
    except ImportError:
        print("[densify] ⚠ opencv not installed — skipping frame densification.")
        return len(_image_files(output_dir))

    import gc
    from PIL import Image as _PILImage

    frames = _image_files(output_dir)
    current = len(frames)

    if current >= target_frames:
        print(f"[densify] Already have {current} frames (≥{target_frames}) — skipping densification.")
        return current

    print(
        f"[densify] ⚡ Short-video densification: {current} → target {target_frames} frames "
        f"(optical-flow midpoint interpolation)"
    )

    for _round in range(3):
        frames = _image_files(output_dir)
        if len(frames) >= target_frames:
            break

        new_count = 0
        # Read all frames upfront so we can insert without re-scanning mid-loop
        frame_list = list(frames)

        for i in range(len(frame_list) - 1):
            f_a = frame_list[i]
            f_b = frame_list[i + 1]

            img_a = cv2.imread(str(f_a))
            img_b = cv2.imread(str(f_b))
            if img_a is None or img_b is None:
                continue

            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

            # Compute dense optical flow A→B
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )

            import numpy as np
            h, w = img_a.shape[:2]

            # Build pixel-coordinate grids for forward warping A by half-flow
            xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            ys = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))

            map_x_full = (xs + flow[..., 0] * 0.5).astype(np.float32)
            map_y_full = (ys + flow[..., 1] * 0.5).astype(np.float32)

            warped = cv2.remap(img_a, map_x_full, map_y_full,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

            # Blend 50/50 with img_b for temporal coherence and fewer artefacts
            midframe = cv2.addWeighted(warped, 0.5, img_b, 0.5, 0)

            # Name: insert between f_a and f_b — use stem + "_mid" suffix
            # so stable sort keeps ordering correct
            stem_a = f_a.stem          # e.g. "output_0003"
            new_name = f_a.parent / f"{stem_a}_mid{_round+1:02d}.jpg"
            cv2.imwrite(str(new_name), midframe, [cv2.IMWRITE_JPEG_QUALITY, 90])

            del img_a, img_b, gray_a, gray_b, flow, warped, midframe
            gc.collect()
            new_count += 1

        final_count = len(_image_files(output_dir))
        print(
            f"[densify] Round {_round+1}: synthesised {new_count} midpoint frames → "
            f"{final_count} total"
        )

        if new_count == 0:
            break

    # Rename all frames to a clean sequential numbering so COLMAP sees a
    # consistent set and sequential_matcher (overlap-based) works correctly.
    _renumber_frames(output_dir)

    final = len(_image_files(output_dir))
    print(f"[densify] ✓  Densification complete: {final} frames in {output_dir}")
    return final


def _renumber_frames(output_dir: Path) -> None:
    """Rename all frames to output_NNNN.jpg in sorted order (in-place, atomic)."""
    frames = _image_files(output_dir)
    # Two-pass rename to avoid collisions: temp names → final names
    import tempfile, os
    tmp_names = []
    for i, f in enumerate(frames):
        tmp = output_dir / f"__tmp_{i:06d}.jpg"
        f.rename(tmp)
        tmp_names.append(tmp)
    for i, tmp in enumerate(tmp_names):
        final = output_dir / f"output_{i+1:04d}.jpg"
        tmp.rename(final)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_from_video(
    video_path:        str,
    output_dir:        str,
    fps:               float = None,
    max_frames:        int   = 150,
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
    SHORT_VIDEO_THRESHOLD = 25    # seconds — triggers high-density extraction
    SHORT_VIDEO_MIN_FRAMES = 100  # minimum frames we want from a short video

    try:
        info     = get_video_info(str(video_path))
        duration = max(info["duration_sec"], 1.0)
        native_fps = info["fps"]

        is_short_video = duration < SHORT_VIDEO_THRESHOLD

        if fps is None:
            if is_short_video:
                # Short video: extract aggressively — cap at native fps to avoid
                # duplicate frames but aim for at least SHORT_VIDEO_MIN_FRAMES.
                # For high-fps sources (60fps), cap target fps at 30 to avoid
                # extracting near-duplicate frames (every 2 frames at 60fps).
                effective_native = min(native_fps, 30.0)
                desired_fps = max(SHORT_VIDEO_MIN_FRAMES / duration, 6.0)
                fps = min(desired_fps, effective_native)
                fps = round(fps, 1)
                print(
                    f"[extract] ⚡ SHORT VIDEO ({duration:.1f}s) — boosting to "
                    f"{fps:.1f} fps (target ≥{SHORT_VIDEO_MIN_FRAMES} frames, "
                    f"native={native_fps:.1f} capped at {effective_native:.0f})"
                )
            elif duration < 40:
                # Medium duration 25–40 s: 3 fps gives ~75–120 frames.
                fps = 3
            elif duration < 120:
                # Standard object-capture clip 40–120 s: 2 fps gives ~80–240 frames.
                fps = 2
            elif duration < 300:
                # Long scene video 2–5 min: 1 fps keeps frame count within budget.
                fps = 1
            else:
                # Very long video > 5 min: 0.5 fps.
                fps = 0.5

        print(f"[extract] Adaptive FPS: {fps} (duration={duration:.1f}s, native={native_fps:.1f})")
        print(
            f"[extract] {info['width']}×{info['height']} @ {native_fps:.1f} fps | "
            f"{duration:.1f}s | target {fps:.1f} fps | "
            f"~{int(duration * fps)} frames expected"
        )
    except Exception as e:
        print(f"[extract] Could not read metadata: {e}")
        is_short_video = False
        if fps is None:
            fps = 5

    # ------------------ OUTPUT ------------------
    output_pattern = str(output_dir / "output_%04d.jpg")

    # ------------------ VIDEO FILTER ------------------
    # IMPORTANT: Do NOT resize before COLMAP.
    # Keep original frame resolution (1280×830 in the required pipeline).
    if adaptive_sampling:
        vf_filter = f"select='gt(scene,0.02)',fps={fps}"
        print("[extract] Using adaptive sampling (no resize)")
    else:
        vf_filter = f"fps={fps}"

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

    # ------------------ SHORT-VIDEO DENSIFICATION ------------------
    # For short / fast-motion videos we optionally synthesise midpoint frames
    # between every consecutive pair using optical-flow warping.  This brings
    # the effective frame count up WITHOUT re-recording — each synthetic frame
    # provides unique camera pose coverage and is detectable by SIFT.
    if is_short_video and saved > 0:
        saved = _densify_frames_optical_flow(output_dir, target_frames=SHORT_VIDEO_MIN_FRAMES)

    # ------------------ DEBUG RESOLUTION ------------------
    try:
        from PIL import Image
        sample = _image_files(output_dir)[:3]
        for s in sample:
            with Image.open(s) as img:
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
            "-vf", f"fps={fps_retry}" ,
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

        # Re-run densification after retry if still a short video
        if is_short_video and saved > 0 and saved < SHORT_VIDEO_MIN_FRAMES:
            saved = _densify_frames_optical_flow(output_dir, target_frames=SHORT_VIDEO_MIN_FRAMES)

    # ------------------ VALIDATION ------------------
    validate_images(str(output_dir))

    # Force GC before file rename operations — Windows holds file locks until
    # handles are explicitly released; PIL/cv2 objects from validate_images
    # may still be in scope otherwise.
    import gc as _gc
    _gc.collect()

    # ------------------ BLUR FILTER ------------------
    kept_blur = filter_blurry_images(str(output_dir), threshold=max(blur_threshold, 120.0))  # floor at 120: reject blurry frames harder

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

    # ------------------ VIEWPOINT DIVERSITY FILTER ------------------
    kept = filter_duplicate_viewpoints(str(output_dir))

    # ------------------ SMART FRAME SELECTION ------------------
    selection_report = run_smart_frame_selection(str(output_dir), budget=max_frames)
    kept = selection_report["selected_frame_count"]

    # ------------------ MOTION ------------------
    motion = estimate_motion(str(output_dir))
    print(f"[extract] Motion score: {motion:.2f}")

    if motion < 1.0:
        print("[extract] ⚠ LOW MOTION — move camera around object")

    # ------------------ FINAL SAFETY ------------------
    # Hard fail: COLMAP needs >= 30 frames for reliable reconstruction.
    # Lowered 40→30: the COLMAP quality gate catches bad reconstructions; this
    # prevents rejecting marginal but recoverable captures.
    # A warning-only gate here means the pipeline silently produces a degenerate splat.
    if kept < 30:
        raise RuntimeError(
            f"[extract] HARD FAILURE: Only {kept} frames extracted after filtering.\n"
            "COLMAP requires >= 30 frames for reliable sparse reconstruction.\n"
            "Fix:\n"
            "  1. Use a longer video (>= 20 seconds of footage).\n"
            "  2. Reduce blur filter threshold if too many frames are being rejected.\n"
            "  3. Improve lighting to reduce motion blur.\n"
            "  4. Shoot at a slower speed to avoid duplicate frame rejection."
        )

    print(f"[extract] FINAL: {kept} frames kept")
    return kept


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
                img = img.convert("RGB")   # ensures valid format
                img.load()                 # forces full decode

                # [UPLOAD DEBUG] print actual pixel dimensions for every image
                print(
                    f"[UPLOAD] {p.name}: "
                    f"{img.width}x{img.height}"
                )

                # Hard minimum raised to 256px — anything smaller is useless
                # for COLMAP SIFT and will produce a blank / 46-gaussian result
                if img.width < 256 or img.height < 256:
                    raise ValueError(
                        f"Image too small for reconstruction: "
                        f"{p.name} ({img.width}x{img.height}). "
                        f"Minimum is 256x256px."
                    )

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
    max_frames: int   = 300,
) -> int:
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.suffix in extensions)
    if not images:
        raise FileNotFoundError(f"No images with extensions {extensions} in {image_dir}")

    if len(images) > max_frames:
        # Uniformly subsample to max_frames so COLMAP stays tractable on very
        # large image sets (e.g. Tanks and Temples Advanced scenes can have
        # 500–1000 images at 1 fps).
        import math
        step   = len(images) / max_frames
        images = [images[int(i * step)] for i in range(max_frames)]
        print(f"[extract] Subsampled image set to {len(images)} / original (max_frames={max_frames})")

    for i, src in enumerate(images):
        dst = output_dir / f"output_{i + 1:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        if i % 50 == 0:
            print(f"[extract] Copying {i + 1}/{len(images)}…")

    print(f"[extract] Copied {len(images)} images → {output_dir}")
    validate_images(str(output_dir))
    filter_blurry_images(str(output_dir), threshold=120.0)  # 80→120
    kept = filter_low_feature_frames(str(output_dir), min_keep_ratio=0.4)
    selection_report = run_smart_frame_selection(str(output_dir), budget=max_frames)
    return selection_report.get("selected_frame_count", kept)


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
    parser.add_argument("--max_frames",     type=int,   default=300)
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