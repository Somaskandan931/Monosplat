"""
config_loader.py — Central default configuration for MonoSplat training.

IMPORTANT: _DEFAULTS must always mirror the values in configs/config.yaml.
_DEFAULTS are ONLY used as last-resort fallbacks when a key is absent from
config.yaml.  A mismatch causes silent degraded training behaviour.

FIXES APPLIED:
  [FIX-1] _DEFAULTS now includes 'data' and 'colmap' sections so that
          prepare_dataset.py gets correct values even when config.yaml
          is missing those sections (backward compatibility).
  [FIX-2] densify_grad_threshold default: 0.00002 → 0.0003 (matches config.yaml).
  [FIX-3] densification_interval default: 100 → 200 (matches config.yaml).
  [FIX-4] lambda_dssim default: 0.4 → 0.2 (original 3DGS paper value).
  [FIX-5] load_config() always returns a plain dict — prepare_dataset.py
          must access sections as cfg["colmap"]["quality"], NOT cfg.colmap.quality.
          Added _cfg_get() helper for safe attribute-style access used in
          older prepare_dataset.py that does getattr(cfg.colmap, "quality").
  [FIX-6] All numeric defaults updated to match config.yaml exactly:
          iterations 18000→30000, max_gaussians 80000→150000,
          densify_until_iter 5000→15000, lambda_lpips 0.0→0.05,
          opacity_reset_interval 3000→1000, position_lr_max_steps 18000→30000.
          renderer.max_gaussians 80000→150000 (must equal training.max_gaussians).
"""

import copy
from pathlib import Path
from typing import Any, Dict, Union

import yaml

# ---------------------------------------------------------------------------
# Hardcoded defaults — aligned with config.yaml and standard 3DGS paper values
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "training": {
        # ── Mirrors configs/config.yaml exactly ──────────────────────────────
        # If you change a value in config.yaml, update it here too.
        # These values are ONLY used when the key is absent from config.yaml.
        "iterations": 30000,                 # FIX-6: was 18000
        "densify_from_iter": 500,
        "densify_until_iter": 15000,         # FIX-6: was 5000
        "densification_interval": 200,       # FIX-3: was 100; matches config.yaml
        "densify_grad_threshold": 0.0003,    # FIX-2: was 0.00002; matches config.yaml
        "max_gaussians": 150000,             # FIX-6: was 80000
        "opacity_reset_interval": 1000,      # FIX-6: was 3000
        "lambda_opacity_reg": 0.001,
        "lambda_dssim": 0.2,                 # FIX-4
        "lambda_lpips": 0.05,                # FIX-6: was 0.0
        "percent_dense": 0.01,
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,      # FIX-6: was 18000
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "save_iterations": [1000, 5000, 10000, 20000, 30000],
        "checkpoint_iterations": [1000, 5000, 10000, 20000, 30000],
        "test_iterations": [5000, 10000, 20000, 30000],
    },
    "model": {
        "sh_degree": 3,
    },
    "renderer": {
        # FIX-6: must equal training.max_gaussians; renderer was silently
        # capping exported Gaussians to 80k while training used 150k.
        "max_gaussians": 150000,             # FIX-6: was 80000
    },
    "pipeline": {
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "debug": False,
    },
    # FIX-1: Added missing sections for prepare_dataset.py
    "data": {
        "max_frames": 300,
        "image_width": 0,
        "image_height": 0,
    },
    "colmap": {
        "quality": "medium",
        "binary_path": "colmap",
        "camera_model": "OPENCV",
        "single_camera": True,
    },
}


# ---------------------------------------------------------------------------
# Tiny helper: safe attr-style access on a nested dict
# Used by prepare_dataset.py: getattr(cfg["colmap"], "quality", "medium")
# After FIX-1 prepare_dataset.py should use cfg["colmap"]["quality"] directly,
# but this shim keeps old code working without changes.
# ---------------------------------------------------------------------------

class _DictProxy:
    """
    Wraps a plain dict so that attribute access works like dict access.
    getattr(proxy, "quality", "medium") → dict.get("quality", "medium")
    """
    def __init__(self, d: dict):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, key):
        d = object.__getattribute__(self, "_d")
        if key in d:
            return d[key]
        raise AttributeError(f"Config section has no key '{key}'")

    def get(self, key, default=None):
        return object.__getattribute__(self, "_d").get(key, default)

    def __contains__(self, key):
        return key in object.__getattribute__(self, "_d")

    def __getitem__(self, key):
        return object.__getattribute__(self, "_d")[key]

    def __repr__(self):
        return repr(object.__getattribute__(self, "_d"))


class _ConfigProxy(dict):
    """
    A dict subclass where top-level sections are also accessible as attributes
    returning _DictProxy wrappers. This makes both
        cfg["colmap"]["quality"]   AND
        cfg.colmap.quality          AND
        getattr(cfg.colmap, "quality", default)
    all work correctly regardless of which style prepare_dataset.py uses.
    """
    def __getattr__(self, key):
        if key in self:
            val = self[key]
            if isinstance(val, dict):
                return _DictProxy(val)
            return val
        raise AttributeError(f"Config has no section '{key}'")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_defaults() -> "_ConfigProxy":
    """Return a deep copy of the default config (safe to mutate)."""
    return _ConfigProxy(copy.deepcopy(_DEFAULTS))


def load_config(path: Union[str, Path]) -> "_ConfigProxy":
    """Alias for load_yaml."""
    return load_yaml(path)


def load_yaml(path: Union[str, Path]) -> "_ConfigProxy":
    """Load a YAML config file and deep-merge it over _DEFAULTS."""
    cfg = get_defaults()
    with open(path, "r") as fh:
        overrides = yaml.safe_load(fh) or {}
    _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* in-place."""
    for key, val in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val