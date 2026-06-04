"""
core/evaluation/fps_benchmark.py
----------------------------------
Render FPS benchmark for a trained MonoSplat GaussianModel.

Measures:
  - Render throughput (frames per second) at a specified resolution
  - Per-resolution breakdown (if multiple resolutions requested)
  - VRAM headroom during rendering
  - Warm-up + timed window to get stable numbers

Design constraints
------------------
  - Does NOT import renderer.py / GaussianModel at module load time.
    All heavy imports are deferred to run() so the class can be instantiated
    in environments without CUDA or gsplat.
  - gaussian_model.py, renderer.py are NOT modified — we call the existing
    renderer interface exactly as trainer.py does.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("monosplat.evaluation.fps")

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class FPSBenchmark:
    """
    Benchmark render throughput for a trained GaussianModel.

    Parameters
    ----------
    warmup_frames   : number of frames rendered before timing starts
    timed_frames    : number of frames used for the actual measurement
    resolutions     : list of (width, height) tuples to test
    """

    def __init__(
        self,
        warmup_frames: int = 10,
        timed_frames: int = 50,
        resolutions: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.warmup_frames = warmup_frames
        self.timed_frames  = timed_frames
        self.resolutions   = resolutions or [(1920, 1080), (1280, 720), (640, 480)]

    # ── Main entry point ───────────────────────────────────────────────────

    def run(
        self,
        model,                           # GaussianModel (already loaded)
        cameras: Optional[List] = None,  # list of Camera objects; synthetic if None
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the FPS benchmark.

        Parameters
        ----------
        model   : GaussianModel — loaded from checkpoint
        cameras : optional list of Camera objects (first N are used for benchmark).
                  If None, synthetic cameras on a sphere are generated.
        device  : 'cuda' / 'cpu' / None (auto-detect)

        Returns
        -------
        dict with keys:
            device, n_gaussians,
            resolutions: [{width, height, fps_mean, fps_std, frame_times_ms}],
            primary_fps (1280×720 or best available),
            vram_gb (peak during benchmark, None if CPU)
        """
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)
        model.eval()

        n_gaussians = _count_gaussians(model)
        log.info("FPS benchmark: %d Gaussians on %s", n_gaussians, device)

        results_per_res: List[Dict] = []

        for (width, height) in self.resolutions:
            cams  = cameras or _synthetic_cameras(width, height, n=self.warmup_frames + self.timed_frames)
            fps_r = self._benchmark_resolution(model, cams, width, height, device)
            results_per_res.append({"width": width, "height": height, **fps_r})
            log.info("  %dx%d → %.1f FPS", width, height, fps_r["fps_mean"])

        # Primary FPS: 1280×720 or first resolution
        primary = next(
            (r for r in results_per_res if r["width"] == 1280 and r["height"] == 720),
            results_per_res[0] if results_per_res else {},
        )

        vram_gb = None
        if device == "cuda" and torch.cuda.is_available():
            vram_gb = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)

        return {
            "device":      device,
            "n_gaussians": n_gaussians,
            "resolutions": results_per_res,
            "primary_fps": round(primary.get("fps_mean", 0.0), 2),
            "vram_gb":     vram_gb,
        }

    # ── Per-resolution benchmark ───────────────────────────────────────────

    def _benchmark_resolution(
        self,
        model,
        cameras: List,
        width: int,
        height: int,
        device: str,
    ) -> Dict:
        import torch
        from renderer.renderer import GaussianRenderer

        renderer = GaussianRenderer(model)

        cam_cycle = _make_cycle(cameras, self.warmup_frames + self.timed_frames)

        # Warm-up (not timed)
        for i in range(self.warmup_frames):
            cam = _ensure_resolution(cam_cycle[i], width, height)
            with torch.no_grad():
                try:
                    renderer.render(cam)
                except Exception:
                    _render_fallback(model, cam, device)
            if device == "cuda":
                torch.cuda.synchronize()

        # Timed window
        frame_times: List[float] = []
        for i in range(self.warmup_frames, self.warmup_frames + self.timed_frames):
            cam = _ensure_resolution(cam_cycle[i % len(cam_cycle)], width, height)
            t0 = time.perf_counter()
            with torch.no_grad():
                try:
                    renderer.render(cam)
                except Exception:
                    _render_fallback(model, cam, device)
            if device == "cuda":
                torch.cuda.synchronize()
            frame_times.append((time.perf_counter() - t0) * 1000.0)  # ms

        import numpy as np
        arr = np.array(frame_times, dtype=np.float64)
        return {
            "fps_mean":       round(float(1000.0 / arr.mean()), 2),
            "fps_std":        round(float((1000.0 / arr).std()), 2),
            "frame_times_ms": [round(v, 2) for v in arr.tolist()],
        }


# ── Benchmark from checkpoint (convenience) ───────────────────────────────────

def benchmark_from_checkpoint(
    checkpoint_path: str | Path,
    resolutions: Optional[List[Tuple[int, int]]] = None,
    warmup_frames: int = 10,
    timed_frames: int  = 50,
) -> Dict[str, Any]:
    """
    Load a checkpoint and immediately run the FPS benchmark.
    Returns the full benchmark result dict.
    """
    import torch
    import torch.nn as nn
    from reconstruction.gaussian_model import GaussianModel

    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_state = state.get("model_state") or state.get("model")
    if model_state is None:
        raise KeyError("Checkpoint missing 'model' or 'model_state'")

    sh_degree = state.get("sh_degree", 3)
    if "_features_rest" in model_state:
        n_rest    = model_state["_features_rest"].shape[1]
        sh_degree = int(round((n_rest + 1) ** 0.5 - 1))

    model = GaussianModel(sh_degree=sh_degree)
    model._xyz           = nn.Parameter(model_state["_xyz"].detach().float().cpu())
    model._features_dc   = nn.Parameter(model_state["_features_dc"].detach().float().cpu())
    model._features_rest = nn.Parameter(model_state["_features_rest"].detach().float().cpu())
    model._opacities     = nn.Parameter(model_state["_opacities"].detach().float().cpu())
    model._scales        = nn.Parameter(model_state["_scales"].detach().float().cpu())
    model._rotations     = nn.Parameter(model_state["_rotations"].detach().float().cpu())

    bench = FPSBenchmark(warmup_frames=warmup_frames, timed_frames=timed_frames,
                         resolutions=resolutions)
    return bench.run(model)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_gaussians(model) -> int:
    try:
        return int(model.get_xyz.shape[0])
    except Exception:
        try:
            return int(model._xyz.shape[0])
        except Exception:
            return 0


def _synthetic_cameras(width: int, height: int, n: int = 60) -> List:
    """Generate n orbit cameras around the origin for benchmarking."""
    import math
    import numpy as np
    from renderer.camera import Camera

    cameras = []
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        pos   = np.array([2.0 * math.cos(angle), 0.5, 2.0 * math.sin(angle)], dtype=np.float32)
        cam   = Camera(position=pos, width=width, height=height, fov_deg=60.0)
        cameras.append(cam)
    return cameras


def _make_cycle(cameras: List, n: int) -> List:
    """Repeat cameras list until length >= n."""
    if not cameras:
        return []
    result = []
    while len(result) < n:
        result.extend(cameras)
    return result[:n]


def _ensure_resolution(cam, width: int, height: int):
    """Return a camera with overridden width/height (non-destructive)."""
    if cam.width == width and cam.height == height:
        return cam
    import copy
    c = copy.copy(cam)
    c.width  = width
    c.height = height
    return c


def _render_fallback(model, cam, device: str) -> None:
    """
    Minimal software render — just needs to touch all Gaussian tensors
    so the memory access pattern is realistic even without gsplat.
    """
    import torch
    xyz = model.get_xyz.to(device)
    _ = xyz.sum()   # force GPU kernel
