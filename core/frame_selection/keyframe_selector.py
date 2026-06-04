"""Keyframe selection with budgeted temporal coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .frame_ranker import FrameScore


def select_keyframes(ranked: Iterable[FrameScore], budget: int) -> List[FrameScore]:
    """Select top-ranked frames while preserving temporal coverage."""
    ranked = list(ranked)
    if len(ranked) <= budget:
        return sorted(ranked, key=lambda item: item.index)

    selected: dict[Path, FrameScore] = {}

    # Anchor the sequence so COLMAP keeps endpoints and broad trajectory hints.
    selected[ranked[0].path] = ranked[0]
    selected[ranked[-1].path] = ranked[-1]

    # Reserve roughly 30% for uniform temporal coverage.
    temporal_slots = max(0, min(budget - len(selected), int(budget * 0.30)))
    if temporal_slots:
        step = len(ranked) / temporal_slots
        for i in range(temporal_slots):
            candidate = ranked[min(len(ranked) - 1, int(i * step))]
            selected[candidate.path] = candidate
            if len(selected) >= budget:
                break

    for candidate in sorted(ranked, key=lambda item: item.score, reverse=True):
        selected[candidate.path] = candidate
        if len(selected) >= budget:
            break

    return sorted(selected.values(), key=lambda item: item.index)
