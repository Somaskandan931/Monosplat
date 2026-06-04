"""Budgeted smart frame selection engine."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Set

from core.dataset_analysis.common import image_files
from .duplicate_remover import find_duplicate_frames, move_unselected_frames
from .frame_ranker import rank_frames
from .keyframe_selector import select_keyframes


FRAME_BUDGETS = (100, 150, 200, 300)


@dataclass
class SmartFrameSelectionEngine:
    """Rank and select frames before COLMAP."""

    budgets: tuple[int, ...] = FRAME_BUDGETS

    def select(
        self,
        image_dir: str | Path,
        budget: int = 150,
        output_path: Optional[str | Path] = None,
        mutate: bool = True,
    ) -> Dict:
        image_dir = Path(image_dir)
        original_count = len(image_files(image_dir))
        resolved_budget = self.resolve_budget(budget)
        duplicate_report = find_duplicate_frames(image_dir)
        ranked = rank_frames(image_dir)
        selected = select_keyframes(ranked, min(resolved_budget, len(ranked)))
        selected_paths: Set[Path] = {item.path for item in selected}

        moved = 0
        if mutate and len(selected_paths) < original_count:
            moved = move_unselected_frames(image_dir, selected_paths)

        report = {
            "generated_at": time.time(),
            "image_dir": str(image_dir),
            "requested_budget": int(budget),
            "resolved_budget": int(resolved_budget),
            "original_frame_count": int(original_count),
            "selected_frame_count": int(len(selected)),
            "rejected_frame_count": int(moved if mutate else max(0, original_count - len(selected))),
            "duplicate_ratio": duplicate_report["duplicate_ratio"],
            "selected_frames": [item.path.name for item in selected],
            "ranked_frames": [
                {
                    **asdict(item),
                    "path": item.path.name,
                }
                for item in sorted(ranked, key=lambda frame: frame.score, reverse=True)
            ],
        }
        if output_path is not None:
            save_frame_selection_report(report, output_path)
        return report

    def resolve_budget(self, budget: int) -> int:
        """Resolve arbitrary input to the nearest supported budget at or above it."""
        for candidate in self.budgets:
            if budget <= candidate:
                return candidate
        return self.budgets[-1]


def save_frame_selection_report(report: Dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    tmp.replace(path)
    return path
