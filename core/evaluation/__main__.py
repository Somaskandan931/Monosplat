"""
core/evaluation/__main__.py
-----------------------------
Standalone CLI for MonoSplat Evaluation Framework.

Usage examples
--------------
# Evaluate a single run from its checkpoint:
python -m core.evaluation \\
    --run-dir experiments/run_20260603_161200 \\
    --checkpoint experiments/run_20260603_161200/checkpoints/checkpoint_015000.ckpt \\
    --image-dir data/frames \\
    --skip-fps

# Compare two runs:
python -m core.evaluation compare \\
    experiments/run_a \\
    experiments/run_b \\
    --labels "Baseline" "Phase8" \\
    --output-dir experiments/comparisons
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Ensure repo is importable ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from core.evaluation import EvaluationPipeline

    run_dir = Path(args.run_dir)
    run_id  = run_dir.name

    training_result: dict = {}
    if args.checkpoint:
        training_result["checkpoint_path"] = args.checkpoint
    if args.model_path:
        training_result["model_path"] = args.model_path

    pipeline = EvaluationPipeline(
        run_dir=run_dir,
        run_id=run_id,
        device=args.device or None,
    )

    # If checkpoint is provided, load model for FPS benchmark
    model = None
    if args.checkpoint and not args.skip_fps:
        try:
            import torch
            import torch.nn as nn
            from reconstruction.gaussian_model import GaussianModel
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            ms    = state.get("model_state") or state.get("model")
            if ms:
                sh = state.get("sh_degree", 3)
                if "_features_rest" in ms:
                    n_rest = ms["_features_rest"].shape[1]
                    sh     = int(round((n_rest + 1) ** 0.5 - 1))
                model = GaussianModel(sh_degree=sh)
                model._xyz           = nn.Parameter(ms["_xyz"].detach().float().cpu())
                model._features_dc   = nn.Parameter(ms["_features_dc"].detach().float().cpu())
                model._features_rest = nn.Parameter(ms["_features_rest"].detach().float().cpu())
                model._opacities     = nn.Parameter(ms["_opacities"].detach().float().cpu())
                model._scales        = nn.Parameter(ms["_scales"].detach().float().cpu())
                model._rotations     = nn.Parameter(ms["_rotations"].detach().float().cpu())
        except Exception as exc:
            print(f"[warn] Could not load model for FPS benchmark: {exc}")

    paths = pipeline.run_post_training(
        model=model,
        training_result=training_result,
        dataset_path=args.image_dir,
        skip_fps=args.skip_fps or model is None,
    )

    print("\nEvaluation complete:")
    for fmt, path in paths.items():
        print(f"  {fmt:4s} → {path}")


def _cmd_compare(args: argparse.Namespace) -> None:
    from core.evaluation import compare_runs
    result = compare_runs(
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
        labels=args.labels or None,
    )
    print("\nComparison complete:")
    print(json.dumps(result.get("winner", {}), indent=2))
    if result.get("html_path"):
        print(f"\nHTML → {result['html_path']}")


# ── Argument parser ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m core.evaluation",
        description="MonoSplat Evaluation Framework CLI",
    )
    sub = p.add_subparsers(dest="command")

    # ── evaluate (default) ────────────────────────────────────────────────
    ev = sub.add_parser("evaluate", help="Evaluate a single run")
    ev.add_argument("--run-dir",     required=True, help="Experiment directory")
    ev.add_argument("--checkpoint",  default=None,  help="Path to .ckpt file")
    ev.add_argument("--model-path",  default=None,  help="Path to final.ply")
    ev.add_argument("--image-dir",   default=None,  help="Path to frames directory")
    ev.add_argument("--device",      default=None,  help="cuda / cpu")
    ev.add_argument("--skip-fps",    action="store_true", help="Skip FPS benchmark")

    # ── compare ───────────────────────────────────────────────────────────
    cmp = sub.add_parser("compare", help="Compare multiple runs")
    cmp.add_argument("run_dirs", nargs="+", help="Two or more experiment directories")
    cmp.add_argument("--labels",     nargs="+", default=None)
    cmp.add_argument("--output-dir", default=None)

    return p


def main() -> None:
    parser = _build_parser()
    # Default to 'evaluate' if first arg looks like a path
    argv = sys.argv[1:]
    if argv and argv[0] not in ("evaluate", "compare", "-h", "--help"):
        argv = ["evaluate"] + argv

    args = parser.parse_args(argv)
    if args.command == "compare":
        _cmd_compare(args)
    else:
        _cmd_evaluate(args)


if __name__ == "__main__":
    main()
