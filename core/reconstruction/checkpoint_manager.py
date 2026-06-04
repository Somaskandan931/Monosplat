"""Checkpoint discovery, validation, and resume helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional


class CheckpointManager:
    def __init__(self, *roots: str | Path) -> None:
        self.roots = [Path(root) for root in roots if root]

    def latest_checkpoint(self) -> Optional[Path]:
        candidates = []
        for root in self.roots:
            if root.is_file() and root.suffix == ".ckpt":
                candidates.append(root)
            elif root.exists():
                candidates.extend(root.rglob("*.ckpt"))
        valid = [path for path in candidates if self.validate_checkpoint(path)["valid"]]
        if not valid:
            return None
        return max(valid, key=lambda path: path.stat().st_mtime)

    def validate_checkpoint(self, checkpoint_path: str | Path) -> Dict:
        path = Path(checkpoint_path)
        result = {"path": str(path), "valid": False, "reason": None, "iteration": None}
        if not path.exists():
            result["reason"] = "missing"
            return result
        if path.stat().st_size <= 1024:
            result["reason"] = "too_small"
            return result
        try:
            import torch
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            if not isinstance(state, dict) or not ("model" in state or "model_state" in state):
                result["reason"] = "missing_model_state"
                return result
            result["iteration"] = state.get("iteration")
        except Exception as exc:
            result["reason"] = str(exc)
            return result
        result["valid"] = True
        return result

    def auto_resume_path(self, explicit_resume: str | Path | None = None) -> Optional[Path]:
        if explicit_resume:
            path = Path(explicit_resume)
            return path if self.validate_checkpoint(path)["valid"] else None
        return self.latest_checkpoint()

    def mirror_checkpoint(self, checkpoint_path: str | Path, target_dir: str | Path) -> Optional[Path]:
        src = Path(checkpoint_path)
        if not src.exists():
            return None
        target = Path(target_dir) / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        return target

    def write_resume_report(self, path: str | Path, resume_path: Optional[str | Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"resume_checkpoint": str(resume_path) if resume_path else None}
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(target)
        return target
