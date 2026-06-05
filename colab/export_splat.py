"""
scripts/export_splat.py
-----------------------
Export a trained Gaussian model checkpoint to viewer-compatible formats.

Outputs:
    final.ply    — Standard 3DGS PLY (MeshLab, CloudCompare, SuperSplat)
    final.splat  — Compact binary (antimatter15 viewer, SuperSplat, Three.js)

Usage:
    python scripts/export_splat.py --checkpoint outputs/checkpoints/checkpoint_015000.ckpt
    python scripts/export_splat.py --ply outputs/exports/final.ply  # convert existing PLY

FIXES APPLIED:
  [FIX-1] sys.path now inserts _REPO_ROOT/src so that bare module names
          (reconstruction, utils) resolve correctly.  Previously the script
          only added _REPO_ROOT, causing "No module named 'src'" when code
          ran from a directory other than the repo root.
  [FIX-2] All imports changed from 'src.reconstruction.*' / 'src.utils.*'
          to bare 'reconstruction.*' / 'utils.*' — consistent with how
          train.py, prepare_dataset.py, and trainer.py import them.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC       = _REPO_ROOT / "src"

# FIX-1: add src/ to sys.path (bare module names)
for _p in (str(_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_from_state(state: dict):
    """Reconstruct a GaussianModel from a checkpoint state dict."""
    import torch.nn as nn
    # FIX-2: bare import — was 'from src.reconstruction.gaussian_model'
    from reconstruction.gaussian_model import GaussianModel

    model_state = state.get("model_state") or state.get("model")
    if model_state is None:
        raise KeyError("Checkpoint does not contain 'model' or 'model_state'")

    # Infer SH degree from feature_rest shape
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
    return model


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def from_checkpoint(ckpt_path: str, output_dir: str) -> None:
    import torch
    # FIX-2: bare imports
    from utils.io_utils import save_ply, save_splat

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model     = _load_model_from_state(state)
    gaussians = model.get_state()

    # ── Clamp before export ──────────────────────────────────────────────────
    # Prevents giant/invisible splats in the viewer caused by unbounded
    # opacity or scale values that escape [0,1] / sane world-space ranges
    # during training edge cases (NaN recovery, AMP rounding, etc.).
    import numpy as np
    gaussians["opacities"] = np.clip(gaussians["opacities"], 0.0, 1.0)
    # NOTE: do not clamp scales here. GaussianModel.get_state already
    # applies the internal scaling ceiling driven by cameras_extent.
    # Clamping exported scales to an absolute [1e-4, 0.1] breaks runs where
    # scenes are intentionally not normalized (e.g. --no_normalize).


    save_ply(str(out / "final.ply"),     gaussians)
    save_splat(str(out / "final.splat"), gaussians)

    n = len(gaussians["positions"])
    print(f"\nExported {n:,} Gaussians -> {out}")
    print("   final.ply   -> MeshLab / CloudCompare / SuperSplat")
    print("   final.splat -> https://antimatter15.com/splat/")


def from_ply(ply_path: str, output_dir: str) -> None:
    # FIX-2: bare imports
    from utils.io_utils import load_ply, save_splat

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gaussians = load_ply(ply_path)
    save_splat(str(out / "final.splat"), gaussians)

    n = len(gaussians["positions"])
    print(f"\nConverted {n:,} Gaussians -> {out / 'final.splat'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Export Gaussian model to .ply and .splat")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--checkpoint", help="Path to .ckpt training checkpoint")
    grp.add_argument("--ply",        help="Path to existing .ply file to convert to .splat")
    p.add_argument("--output", default="outputs/exports", help="Output directory")
    args = p.parse_args()

    if args.checkpoint:
        from_checkpoint(args.checkpoint, args.output)
    else:
        from_ply(args.ply, args.output)


if __name__ == "__main__":
    main()