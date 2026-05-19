"""
configs/schema.py
=================
Centralised configuration schema for MonoSplat using Pydantic dataclasses.

All thresholds, intervals, and paths are defined here — NO hardcoded values
anywhere else in the codebase.

Usage
-----
    from configs.schema import load_config, MonoSplatConfig

    cfg = load_config("config/config.yaml")
    print(cfg.colmap.min_sparse_points)       # 500
    print(cfg.training.densify_grad_threshold) # 0.0002

Design decisions
----------------
- Pydantic v2 model_validator used for cross-field validation.
- All numeric thresholds documented with the reasoning for their default value.
- Fields that were previously hardcoded in-place are now named constants here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    from pydantic import BaseModel, field_validator, model_validator
    from pydantic import ConfigDict
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback for environments without pydantic
# ---------------------------------------------------------------------------

if not _PYDANTIC_AVAILABLE:
    # Simple dataclass fallback — no validation, but at least typed
    from dataclasses import dataclass, field

    @dataclass
    class ReconstructionConfig:
        min_sparse_points: int = 500
        min_registered_ratio: float = 0.60
        min_frames: int = 40
        quality: str = "medium"
        camera_model: str = "OPENCV"
        single_camera: bool = True
        binary_path: str = "colmap"

    @dataclass
    class LearningRateConfig:
        position: float = 0.00016
        feature: float = 0.0025
        opacity: float = 0.05
        scaling: float = 0.005
        rotation: float = 0.001
        position_final: float = 0.0000016

    @dataclass
    class TrainingConfig:
        iterations: int = 15000
        iterations_cpu: int = 1000
        save_every: int = 5000
        eval_every: int = 1000
        densify_from_iter: int = 500
        densify_until_iter: int = 15000
        densification_interval: int = 100
        opacity_reset_interval: int = 3000
        percent_dense: float = 0.01
        densify_grad_threshold: float = 0.0002
        lambda_dssim: float = 0.2
        output_dir: str = "models/gaussian"
        checkpoint_dir: str = "models/checkpoints"
        learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
        # Scale clamp bounds — prevent divergence on large/small scenes
        log_scale_min: float = -4.0
        log_scale_max: float = 0.5

    @dataclass
    class RendererConfig:
        background_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
        sh_degree: int = 3
        max_gaussians: int = 1_000_000
        batch_size: int = 5000

    @dataclass
    class MonoSplatConfig:
        colmap: ReconstructionConfig = field(default_factory=ReconstructionConfig)
        training: TrainingConfig = field(default_factory=TrainingConfig)
        renderer: RendererConfig = field(default_factory=RendererConfig)


    def load_config(path: str = "config/config.yaml") -> MonoSplatConfig:
        import yaml
        from dataclasses import replace

        with open(path) as f:
            raw = yaml.safe_load(f)

        cfg = MonoSplatConfig()
        # Shallow merge — good enough without pydantic
        if "colmap" in raw:
            cfg.colmap = ReconstructionConfig(**{
                k: v for k, v in raw["colmap"].items()
                if hasattr(cfg.colmap, k)
            })
        if "training" in raw:
            lr_raw = raw["training"].pop("learning_rate", {})
            lr = LearningRateConfig(**{k: v for k, v in lr_raw.items() if hasattr(LearningRateConfig(), k)})
            cfg.training = TrainingConfig(**{
                k: v for k, v in raw["training"].items()
                if hasattr(cfg.training, k)
            })
            cfg.training.learning_rate = lr
        return cfg


else:
    # -----------------------------------------------------------------------
    # Full Pydantic v2 schema
    # -----------------------------------------------------------------------

    class ReconstructionConfig(BaseModel):
        model_config = ConfigDict(validate_assignment=True)

        binary_path: str = "colmap"
        quality: str = "medium"
        camera_model: str = "OPENCV"
        single_camera: bool = True

        # Hard gate: raise RuntimeError if sparse_points < this value.
        # Default 500 matches the "46-Gaussian" root-cause analysis:
        # anything below 500 points produces a degenerate splat.
        min_sparse_points: int = 500

        # Hard gate: minimum registration ratio before pipeline aborts.
        # 0.60 = 60% of frames must be registered by COLMAP.
        min_registered_ratio: float = 0.60

        # Hard gate: minimum frames after filtering before COLMAP is invoked.
        # COLMAP cannot reconstruct from < 40 frames with typical orbit capture.
        min_frames: int = 40

        @field_validator("quality")
        @classmethod
        def validate_quality(cls, v: str) -> str:
            if v not in ("low", "medium", "high"):
                raise ValueError(f"quality must be low/medium/high, got: {v!r}")
            return v

        @field_validator("min_registered_ratio")
        @classmethod
        def validate_ratio(cls, v: float) -> float:
            if not 0.0 < v <= 1.0:
                raise ValueError(f"min_registered_ratio must be in (0, 1], got: {v}")
            return v


    class LearningRateConfig(BaseModel):
        position: float = 0.00016
        feature: float = 0.0025
        opacity: float = 0.05
        scaling: float = 0.005
        rotation: float = 0.001
        position_final: float = 0.0000016   # exponential decay target (100× reduction)


    class TrainingConfig(BaseModel):
        model_config = ConfigDict(validate_assignment=True)

        iterations: int = 15000
        iterations_cpu: int = 1000      # hard cap without GPU
        save_every: int = 5000
        eval_every: int = 1000
        output_dir: str = "models/gaussian"
        checkpoint_dir: str = "models/checkpoints"

        # Densification schedule
        densify_from_iter: int = 500
        densify_until_iter: int = 15000
        densification_interval: int = 100
        opacity_reset_interval: int = 3000

        # Gaussian control
        percent_dense: float = 0.01
        densify_grad_threshold: float = 0.0002
        lambda_dssim: float = 0.2

        # Scale initialisation clamp (prevents divergence on large/small scenes)
        # log_scale_min=-4.0 → min scale ~0.018 world units
        # log_scale_max=0.5  → max scale ~1.65 world units
        log_scale_min: float = -4.0
        log_scale_max: float = 0.5

        learning_rate: LearningRateConfig = LearningRateConfig()

        @model_validator(mode="after")
        def validate_densify_range(self) -> "TrainingConfig":
            if self.densify_from_iter >= self.densify_until_iter:
                raise ValueError(
                    f"densify_from_iter ({self.densify_from_iter}) must be < "
                    f"densify_until_iter ({self.densify_until_iter})"
                )
            return self


    class RendererConfig(BaseModel):
        background_color: List[float] = [1.0, 1.0, 1.0]
        sh_degree: int = 3
        max_gaussians: int = 1_000_000
        batch_size: int = 5000

        @field_validator("sh_degree")
        @classmethod
        def validate_sh(cls, v: int) -> int:
            if v not in (0, 1, 2, 3):
                raise ValueError(f"sh_degree must be 0-3, got: {v}")
            return v


    class MonoSplatConfig(BaseModel):
        model_config = ConfigDict(validate_assignment=True)

        colmap: ReconstructionConfig = ReconstructionConfig()
        training: TrainingConfig = TrainingConfig()
        renderer: RendererConfig = RendererConfig()


    def load_config(path: str = "config/config.yaml") -> MonoSplatConfig:
        """
        Load config.yaml and validate with Pydantic.

        Raises pydantic.ValidationError if any value is out of range,
        which surfaces misconfiguration before the pipeline starts.
        """
        import yaml

        config_path = Path(path)
        if not config_path.exists():
            print(f"[config] {path} not found — using defaults")
            return MonoSplatConfig()

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        # Map legacy config.yaml top-level keys to the schema
        colmap_raw = raw.get("colmap", {})
        training_raw = raw.get("training", {})
        renderer_raw = raw.get("renderer", {})

        # Rename legacy keys
        if "binary_path" not in colmap_raw and "binary_path" in raw.get("colmap", {}):
            colmap_raw["binary_path"] = raw["colmap"]["binary_path"]

        lr_raw = training_raw.pop("learning_rate", {})

        colmap_cfg = ReconstructionConfig(**{
            k: v for k, v in colmap_raw.items()
            if k in ReconstructionConfig.model_fields
        })
        training_cfg = TrainingConfig(
            **{k: v for k, v in training_raw.items() if k in TrainingConfig.model_fields},
            learning_rate=LearningRateConfig(**{
                k: v for k, v in lr_raw.items() if k in LearningRateConfig.model_fields
            }),
        )
        renderer_cfg = RendererConfig(**{
            k: v for k, v in renderer_raw.items()
            if k in RendererConfig.model_fields
        })

        return MonoSplatConfig(
            colmap=colmap_cfg,
            training=training_cfg,
            renderer=renderer_cfg,
        )
