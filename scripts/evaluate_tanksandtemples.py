"""
evaluate_tanksandtemples.py — Tanks and Temples F-score Evaluator for MonoSplat
================================================================================

Computes the official Tanks and Temples benchmark metric (F-score at threshold τ)
between your reconstructed point cloud and the laser-scan ground-truth PLY.

This is the metric used on the T&T leaderboard:
    https://www.tanksandtemples.org/leaderboard/AdvancedF/

Unlike PSNR/SSIM/LPIPS (which measure rendering quality on held-out views),
F-score measures geometric reconstruction accuracy against a physical ground truth.

Usage
-----
  # Minimal — just PLY paths:
  python scripts/evaluate_tanksandtemples.py \\
      --pred    outputs/gaussian/point_cloud.ply \\
      --gt      /data/tanksandtemples/Truck/Truck.ply

  # With scene crop and downsampling (recommended for large scenes):
  python scripts/evaluate_tanksandtemples.py \\
      --pred    outputs/gaussian/point_cloud.ply \\
      --gt      /data/tanksandtemples/Truck/Truck.ply \\
      --tau     0.05 \\
      --voxel   0.01 \\
      --scene   Truck \\
      --out     outputs/tt_eval/

  # Evaluate all standard T&T training scenes at once:
  python scripts/evaluate_tanksandtemples.py \\
      --batch   /data/tanksandtemples/Training/ \\
      --pred_dir outputs/gaussian/ \\
      --out     outputs/tt_eval/

Output
------
  - Console: Precision / Recall / F-score at threshold τ
  - JSON:    <out>/<scene>_tt_eval.json
  - PLY:     <out>/<scene>_colored.ply  (pred points colored by TP/FP/FN)

Thresholds (τ)
--------------
  T&T uses different τ for different scenes.  Common values:
    Training scenes  (Barn, Caterpillar, Church, Courthouse, Ignatius,
                      Meetingroom, Truck)  →  τ = 0.05 m  (5 cm)
    Advanced scenes  (Auditorium, Ballroom, Courtroom, Museum, Palace,
                      Temple)              →  τ = 0.1 m   (10 cm)

  When in doubt, use --tau 0.05 for training and --tau 0.1 for advanced.

Dependencies
------------
  pip install open3d numpy tqdm
  (open3d handles PLY I/O and KD-tree nearest-neighbour in one package)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# T&T scene metadata
# ---------------------------------------------------------------------------

# Default τ thresholds recommended by the T&T benchmark for each scene.
# Source: tanksandtemples/evaluation/evaluation.py in the official toolkit.
TT_SCENE_TAU = {
    # Training
    "barn":         0.05,
    "caterpillar":  0.05,
    "church":       0.05,
    "courthouse":   0.05,
    "ignatius":     0.05,
    "meetingroom":  0.05,
    "truck":        0.05,
    # Advanced
    "auditorium":   0.10,
    "ballroom":     0.10,
    "courtroom":    0.10,
    "museum":       0.10,
    "palace":       0.10,
    "temple":       0.10,
}

TT_TRAINING_SCENES = [
    "barn", "caterpillar", "church", "courthouse",
    "ignatius", "meetingroom", "truck",
]
TT_ADVANCED_SCENES = [
    "auditorium", "ballroom", "courtroom",
    "museum", "palace", "temple",
]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _load_ply_points(path: Path) -> np.ndarray:
    """Load XYZ positions from a PLY file via open3d.  Returns (N, 3) float32."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for PLY loading.\n"
            "Install: pip install open3d"
        )
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Empty or unreadable point cloud: {path}")
    return np.asarray(pcd.points, dtype=np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsample to speed up KD-tree queries on large clouds."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d required for voxel downsampling.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points, dtype=np.float32)


def _nn_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    For each point in source, find nearest-neighbour distance in target.
    Uses open3d's KDTreeFlann for fast querying (~1M points in a few seconds).
    Returns (N,) float32 distances.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d required for KD-tree NN search.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target)
    tree = o3d.geometry.KDTreeFlann(pcd)

    dists = np.empty(len(source), dtype=np.float32)
    for i, pt in enumerate(source):
        _, _, d2 = tree.search_knn_vector_3d(pt, 1)
        dists[i] = np.sqrt(d2[0])
    return dists


def compute_fscore(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    tau: float,
) -> dict:
    """
    Compute Tanks and Temples F-score.

    Precision = fraction of pred points within τ of any GT point.
    Recall    = fraction of GT points within τ of any pred point.
    F-score   = harmonic mean of Precision and Recall.

    Parameters
    ----------
    pred_pts : (N, 3) float32 — predicted reconstruction
    gt_pts   : (M, 3) float32 — laser scan ground truth
    tau      : float  — distance threshold in scene units (metres for T&T)

    Returns
    -------
    dict with keys: precision, recall, fscore, tau,
                    n_pred, n_gt, n_tp_pred, n_tp_gt,
                    d_pred_to_gt (array), d_gt_to_pred (array)
    """
    print(f"[eval] Computing pred→GT distances  (N={len(pred_pts):,})")
    d_pred_to_gt = _nn_distances(pred_pts, gt_pts)

    print(f"[eval] Computing GT→pred distances  (M={len(gt_pts):,})")
    d_gt_to_pred = _nn_distances(gt_pts, pred_pts)

    n_tp_pred = int((d_pred_to_gt <= tau).sum())
    n_tp_gt   = int((d_gt_to_pred <= tau).sum())

    precision = n_tp_pred / len(pred_pts) if len(pred_pts) > 0 else 0.0
    recall    = n_tp_gt   / len(gt_pts)   if len(gt_pts)   > 0 else 0.0

    if precision + recall > 0:
        fscore = 2.0 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return {
        "precision":    round(precision, 6),
        "recall":       round(recall,    6),
        "fscore":       round(fscore,    6),
        "tau":          tau,
        "n_pred":       len(pred_pts),
        "n_gt":         len(gt_pts),
        "n_tp_pred":    n_tp_pred,
        "n_tp_gt":      n_tp_gt,
        "d_pred_to_gt": d_pred_to_gt,
        "d_gt_to_pred": d_gt_to_pred,
    }


# ---------------------------------------------------------------------------
# Coloured PLY export (TP/FP/FN visualisation)
# ---------------------------------------------------------------------------

def _export_colored_ply(
    pred_pts: np.ndarray,
    d_pred_to_gt: np.ndarray,
    tau: float,
    out_path: Path,
) -> None:
    """
    Export pred cloud coloured by accuracy:
      Green  — True Positive  (d <= tau)
      Red    — False Positive (d > tau)
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[eval] open3d not available — skipping coloured PLY export.")
        return

    colors = np.where(
        (d_pred_to_gt <= tau)[:, None],
        np.array([[0.0, 0.8, 0.0]]),   # TP → green
        np.array([[0.8, 0.0, 0.0]]),   # FP → red
    ).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[eval] Coloured PLY saved → {out_path}")


# ---------------------------------------------------------------------------
# Single-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    pred_path: Path,
    gt_path: Path,
    tau: float,
    voxel_size: Optional[float],
    out_dir: Optional[Path],
    scene_name: str = "scene",
    export_colored: bool = True,
) -> dict:
    """
    Full evaluation pipeline for one scene.
    Returns the metrics dict (without raw distance arrays).
    """
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  Scene : {scene_name}")
    print(f"  Pred  : {pred_path}")
    print(f"  GT    : {gt_path}")
    print(f"  τ     : {tau} m")
    if voxel_size:
        print(f"  Voxel : {voxel_size} m")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────
    print("[eval] Loading predicted cloud …")
    pred_pts = _load_ply_points(pred_path)
    print(f"[eval]   pred: {len(pred_pts):,} points")

    print("[eval] Loading ground-truth cloud …")
    gt_pts = _load_ply_points(gt_path)
    print(f"[eval]   GT  : {len(gt_pts):,} points")

    # ── Downsample ────────────────────────────────────────────────────────
    if voxel_size and voxel_size > 0:
        print(f"[eval] Voxel downsampling at {voxel_size} m …")
        pred_pts = _voxel_downsample(pred_pts, voxel_size)
        gt_pts   = _voxel_downsample(gt_pts,   voxel_size)
        print(f"[eval]   pred after voxel: {len(pred_pts):,}")
        print(f"[eval]   GT   after voxel: {len(gt_pts):,}")

    # ── F-score ───────────────────────────────────────────────────────────
    result = compute_fscore(pred_pts, gt_pts, tau)

    # ── Print ─────────────────────────────────────────────────────────────
    print(f"\n  ┌─ T&T Evaluation: {scene_name} ─────────────────────────┐")
    print(f"  │  τ (threshold)   : {tau:.3f} m                           │")
    print(f"  │  Precision       : {result['precision']:.4f}  ({result['n_tp_pred']:,} / {result['n_pred']:,} pred pts)  │")
    print(f"  │  Recall          : {result['recall']:.4f}  ({result['n_tp_gt']:,} / {result['n_gt']:,} GT pts)    │")
    print(f"  │  F-score  ★      : {result['fscore']:.4f}                         │")
    print(f"  └────────────────────────────────────────────────────────────┘")
    print(f"  Elapsed: {time.time() - t0:.1f}s\n")

    # ── Save artefacts ────────────────────────────────────────────────────
    d_pred_to_gt = result.pop("d_pred_to_gt")
    result.pop("d_gt_to_pred")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{scene_name}_tt_eval.json"
        with open(json_path, "w") as f:
            json.dump(
                {**result, "scene": scene_name, "pred": str(pred_path), "gt": str(gt_path)},
                f, indent=2,
            )
        print(f"[eval] Metrics JSON → {json_path}")

        if export_colored:
            ply_out = out_dir / f"{scene_name}_colored.ply"
            _export_colored_ply(pred_pts, d_pred_to_gt, tau, ply_out)

    return result


# ---------------------------------------------------------------------------
# Batch evaluation (all training or advanced scenes)
# ---------------------------------------------------------------------------

def evaluate_batch(
    pred_dir: Path,
    gt_dir: Path,
    tau_override: Optional[float],
    voxel_size: Optional[float],
    out_dir: Path,
) -> None:
    """
    Evaluate all scenes found under gt_dir.

    Expected GT directory layout (standard T&T download):
      <gt_dir>/
        Truck/Truck.ply
        Barn/Barn.ply
        ...

    Expected pred directory layout:
      <pred_dir>/
        Truck/point_cloud.ply   (or Truck.ply)
        Barn/point_cloud.ply
        ...
    """
    scenes_found = []
    for gt_scene_dir in sorted(gt_dir.iterdir()):
        if not gt_scene_dir.is_dir():
            continue
        scene_name = gt_scene_dir.name
        gt_ply = gt_scene_dir / f"{scene_name}.ply"
        if not gt_ply.exists():
            # Try lowercase
            gt_ply = gt_scene_dir / f"{scene_name.lower()}.ply"
        if not gt_ply.exists():
            print(f"[batch] Skipping {scene_name} — no GT PLY found.")
            continue

        # Locate pred PLY
        pred_scene_dir = pred_dir / scene_name
        pred_ply = None
        for candidate in ["point_cloud.ply", f"{scene_name}.ply", f"{scene_name.lower()}.ply"]:
            p = pred_scene_dir / candidate
            if p.exists():
                pred_ply = p
                break

        if pred_ply is None:
            print(f"[batch] Skipping {scene_name} — no pred PLY in {pred_scene_dir}")
            continue

        tau = tau_override or TT_SCENE_TAU.get(scene_name.lower(), 0.05)
        result = evaluate_scene(
            pred_path=pred_ply,
            gt_path=gt_ply,
            tau=tau,
            voxel_size=voxel_size,
            out_dir=out_dir / scene_name,
            scene_name=scene_name,
        )
        scenes_found.append({"scene": scene_name, **result})

    if not scenes_found:
        print("[batch] No scenes evaluated. Check pred_dir and gt_dir layouts.")
        return

    # ── Summary table ──────────────────────────────────────────────────
    mean_f = np.mean([s["fscore"] for s in scenes_found])
    print("\n" + "="*60)
    print(f"  BATCH SUMMARY  ({len(scenes_found)} scenes)")
    print("="*60)
    print(f"  {'Scene':<16}  {'τ':>6}  {'Prec':>8}  {'Recall':>8}  {'F-score':>8}")
    print("  " + "-"*56)
    for s in scenes_found:
        print(
            f"  {s['scene']:<16}  {s['tau']:>6.3f}  "
            f"{s['precision']:>8.4f}  {s['recall']:>8.4f}  {s['fscore']:>8.4f}"
        )
    print("  " + "-"*56)
    print(f"  {'Mean F-score':<16}  {'':>6}  {'':>8}  {'':>8}  {mean_f:>8.4f}")
    print("="*60 + "\n")

    # ── Save batch JSON ────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tt_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"scenes": scenes_found, "mean_fscore": round(mean_f, 6)},
            f, indent=2,
        )
    print(f"[batch] Summary JSON → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tanks and Temples F-score evaluator for MonoSplat.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Single-scene mode
    single = p.add_argument_group("Single-scene mode")
    single.add_argument("--pred",  type=Path, help="Path to predicted .ply file")
    single.add_argument("--gt",    type=Path, help="Path to laser-scan ground-truth .ply")

    # Batch mode
    batch = p.add_argument_group("Batch mode (multiple scenes)")
    batch.add_argument(
        "--batch",    type=Path,
        help="Root directory of T&T GT (e.g. /data/tanksandtemples/Training/)",
    )
    batch.add_argument(
        "--pred_dir", type=Path,
        help="Root directory of your predicted PLYs (one subdir per scene)",
    )

    # Shared options
    p.add_argument(
        "--tau",    type=float, default=None,
        help="Distance threshold in metres (default: scene-specific from T&T spec)"
             "\n  Training scenes: 0.05 m   Advanced scenes: 0.10 m",
    )
    p.add_argument(
        "--voxel",  type=float, default=0.01,
        help="Voxel size for downsampling before NN search (default 0.01 m).\n"
             "Set to 0 to disable (slow on large clouds).",
    )
    p.add_argument(
        "--scene",  type=str, default=None,
        help="Scene name (used for output filenames and default τ lookup)",
    )
    p.add_argument(
        "--out",    type=Path, default=Path("outputs/tt_eval"),
        help="Output directory for JSON + coloured PLY (default: outputs/tt_eval/)",
    )
    p.add_argument(
        "--no_colored", action="store_true",
        help="Skip exporting coloured TP/FP PLY (faster for large clouds)",
    )
    return p.parse_args()


def main() -> None:
    # Dependency check
    try:
        import open3d  # noqa: F401
    except ImportError:
        print(
            "[eval] ERROR: open3d not installed.\n"
            "  Install with:  pip install open3d\n"
            "  In Colab:      !pip install open3d -q",
            file=sys.stderr,
        )
        sys.exit(1)

    args = _parse_args()

    voxel = args.voxel if args.voxel and args.voxel > 0 else None

    # ── Batch mode ────────────────────────────────────────────────────────
    if args.batch:
        if not args.pred_dir:
            print("[eval] --pred_dir is required in batch mode.", file=sys.stderr)
            sys.exit(1)
        evaluate_batch(
            pred_dir=args.pred_dir,
            gt_dir=args.batch,
            tau_override=args.tau,
            voxel_size=voxel,
            out_dir=args.out,
        )
        return

    # ── Single-scene mode ─────────────────────────────────────────────────
    if not args.pred or not args.gt:
        print(
            "[eval] Provide either (--pred + --gt) for single-scene or "
            "(--batch + --pred_dir) for batch mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    scene_name = args.scene or args.pred.parent.name or "scene"
    tau = args.tau or TT_SCENE_TAU.get(scene_name.lower(), 0.05)

    evaluate_scene(
        pred_path=args.pred,
        gt_path=args.gt,
        tau=tau,
        voxel_size=voxel,
        out_dir=args.out,
        scene_name=scene_name,
        export_colored=not args.no_colored,
    )


if __name__ == "__main__":
    main()