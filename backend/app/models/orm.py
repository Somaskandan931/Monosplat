"""
backend/app/models/orm.py  (Phase 10 — Project & Database Management)
-----------------------------------------------------------------------
SQLAlchemy ORM models for MonoSplat backend.

Tables (Phase 10 expansion)
---------------------------
users           — registered users / API clients
projects        — named collection of datasets + runs
datasets        — labelled frame-sets attached to a project
experiments     — grouping of training runs under one hypothesis
training_runs   — one end-to-end training execution
models          — finalised, named checkpoints promoted from a run
reports         — evaluation / quality-report records for any entity
jobs            — async background task records (unchanged from Phase 7)
run_metrics     — per-iteration scalar metrics (unchanged from Phase 7)

Phase 10 design notes
---------------------
* Users are lightweight — no password hashing here (auth layer is
  external / JWT-based).  The `api_key_hash` column lets a simple
  HMAC middleware identify callers without storing plaintext keys.
* Projects now carry FK references to their *active* dataset and
  *active* experiment so the UI can default sensibly.
* All JSON columns that were bare dicts in Phase 7 are kept as JSON
  for portability (PostgreSQL will coerce to JSONB automatically via
  the dialect; SQLite stores as TEXT).
* Every table has `created_at` / `updated_at`; soft-delete via
  `deleted_at` (NULL = live row) is optional — columns are omitted
  here to keep the surface small, but the Alembic migration leaves a
  comment marker where they could be added.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from backend.app.database.session import Base


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ══════════════════════════════════════════════════════════════════════════════
# Users
# ══════════════════════════════════════════════════════════════════════════════

class User(Base):
    """
    A MonoSplat user / API client.

    `role` values: 'admin' | 'member' | 'viewer'
    Auth is handled externally (JWT / API-key middleware);
    `api_key_hash` stores HMAC-SHA256(api_key, secret_salt) so we
    can look up a user from a raw key without storing the key itself.
    """
    __tablename__ = "users"

    id           = Column(String(36),  primary_key=True, default=_uuid)
    username     = Column(String(128), nullable=False, unique=True)
    email        = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(255), nullable=True)
    role         = Column(String(32),  nullable=False, default="member")
    api_key_hash = Column(String(128), nullable=True, index=True)
    is_active    = Column(Boolean,     nullable=False, default=True)
    created_at   = Column(DateTime,    nullable=False, default=_now)
    updated_at   = Column(DateTime,    nullable=False, default=_now, onupdate=_now)

    # relationships
    projects = relationship("Project", back_populates="owner",
                            foreign_keys="Project.owner_id")


# ══════════════════════════════════════════════════════════════════════════════
# Projects
# ══════════════════════════════════════════════════════════════════════════════

class Project(Base):
    """
    A named workspace that groups datasets, experiments, runs,
    models, and reports.

    `active_dataset_id` / `active_experiment_id` are convenience
    FK pointers updated by the API when the user selects defaults.
    They are nullable because they are populated *after* child rows
    exist.
    """
    __tablename__ = "projects"

    id                    = Column(String(36),  primary_key=True, default=_uuid)
    owner_id              = Column(String(36),  ForeignKey("users.id"), nullable=True)
    name                  = Column(String(255), nullable=False)
    description           = Column(Text,        nullable=True)
    tags                  = Column(JSON,        nullable=True)   # list[str]
    settings              = Column(JSON,        nullable=True)   # dict of project-level config
    active_dataset_id     = Column(String(36),  ForeignKey("datasets.id",    use_alter=True,
                                                             name="fk_project_active_dataset"),
                                   nullable=True)
    active_experiment_id  = Column(String(36),  ForeignKey("experiments.id", use_alter=True,
                                                             name="fk_project_active_experiment"),
                                   nullable=True)
    created_at            = Column(DateTime,    nullable=False, default=_now)
    updated_at            = Column(DateTime,    nullable=False, default=_now, onupdate=_now)

    # relationships
    owner = relationship(
        "User",
        back_populates="projects",
        foreign_keys=[owner_id]
    )

    datasets = relationship(
        "Dataset",
        back_populates="project",
        foreign_keys="Dataset.project_id",
        cascade="all, delete-orphan"
    )

    experiments = relationship(
        "Experiment",
        back_populates="project",
        foreign_keys="Experiment.project_id",
        cascade="all, delete-orphan"
    )

    training_runs = relationship(
        "TrainingRun",
        back_populates="project",
        cascade="all, delete-orphan"
    )

    models = relationship(
        "Model",
        back_populates="project",
        cascade="all, delete-orphan"
    )

    reports = relationship(
        "Report",
        back_populates="project",
        cascade="all, delete-orphan"
    )

    jobs = relationship(
        "Job",
        back_populates="project"
    )

    active_dataset = relationship(
        "Dataset",
        foreign_keys=[active_dataset_id],
        post_update=True
    )

    active_experiment = relationship(
        "Experiment",
        foreign_keys=[active_experiment_id],
        post_update=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════════════════

class Dataset(Base):
    """
    A labelled set of image frames (or a video source) associated with
    a project.

    `source_type`: 'video' | 'frames' | 'zip'
    `status`:      'uploading' | 'ready' | 'analyzed' | 'error'

    `analysis_report` stores the JSON blob from DatasetAnalysisPipeline.
    `frame_count`, `resolution_w`, `resolution_h` are denormalised
    summaries so the UI doesn't need to parse the full report.
    """
    __tablename__ = "datasets"

    id              = Column(String(36),  primary_key=True, default=_uuid)
    project_id      = Column(String(36),  ForeignKey("projects.id"), nullable=False)
    name            = Column(String(255), nullable=False)
    description     = Column(Text,        nullable=True)
    source_type     = Column(String(32),  nullable=False, default="frames")
    source_path     = Column(Text,        nullable=True)   # server-local path to raw upload
    frames_path     = Column(Text,        nullable=True)   # extracted frames directory
    sparse_path     = Column(Text,        nullable=True)   # COLMAP sparse_text output
    frame_count     = Column(Integer,     nullable=True)
    resolution_w    = Column(Integer,     nullable=True)
    resolution_h    = Column(Integer,     nullable=True)
    size_bytes      = Column(BigInteger,  nullable=True)
    status          = Column(String(32),  nullable=False, default="uploading")
    analysis_report = Column(JSON,        nullable=True)
    quality_score   = Column(Float,       nullable=True)   # 0–100 overall score
    tags            = Column(JSON,        nullable=True)
    created_at      = Column(DateTime,    nullable=False, default=_now)
    updated_at      = Column(DateTime,    nullable=False, default=_now, onupdate=_now)

    # relationships
    project = relationship(
        "Project",
        back_populates="datasets",
        foreign_keys=[project_id]
    )

    training_runs = relationship(
        "TrainingRun",
        back_populates="dataset"
    )

    reports = relationship(
        "Report",
        back_populates="dataset"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Experiments
# ══════════════════════════════════════════════════════════════════════════════

class Experiment(Base):
    """
    A named hypothesis / ablation group that contains one or more
    training runs.

    `base_config` is the YAML config snapshot shared by all runs in the
    experiment; individual runs may override keys via `config_overrides`.
    """
    __tablename__ = "experiments"

    id          = Column(String(36),  primary_key=True, default=_uuid)
    project_id  = Column(String(36),  ForeignKey("projects.id"), nullable=False)
    name        = Column(String(255), nullable=False)
    description = Column(Text,        nullable=True)
    base_config = Column(JSON,        nullable=True)
    tags        = Column(JSON,        nullable=True)
    created_at  = Column(DateTime,    nullable=False, default=_now)
    updated_at  = Column(DateTime,    nullable=False, default=_now, onupdate=_now)

    # relationships
    # relationships
    project = relationship(
        "Project",
        back_populates="experiments",
        foreign_keys=[project_id]
    )

    training_runs = relationship(
        "TrainingRun",
        back_populates="experiment",
        cascade="all, delete-orphan"
    )

    reports = relationship(
        "Report",
        back_populates="experiment"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Training Runs
# ══════════════════════════════════════════════════════════════════════════════

class TrainingRun(Base):
    """
    One end-to-end training execution.  Supersedes the old `Run` model
    from Phase 7 (which is kept as `LegacyRun` during the migration
    window — see Alembic migration 0002).

    `status`: 'pending' | 'running' | 'success' | 'failed' | 'cancelled'

    Metrics timeline is stored in `RunMetric` rows; summary statistics
    are denormalised into `final_metrics` for quick listing queries.

    Exports generated from this run are tracked in `exports` (JSON list
    of {format, path, size_bytes, exported_at}).
    """
    __tablename__ = "training_runs"

    id               = Column(String(36),  primary_key=True, default=_uuid)
    project_id       = Column(String(36),  ForeignKey("projects.id"),    nullable=True)
    experiment_id    = Column(String(36),  ForeignKey("experiments.id"), nullable=True)
    dataset_id       = Column(String(36),  ForeignKey("datasets.id"),    nullable=True)
    run_name         = Column(String(255), nullable=False)
    status           = Column(String(32),  nullable=False, default="pending")
    # paths
    sparse_path      = Column(Text,        nullable=True)
    model_path       = Column(Text,        nullable=True)
    checkpoint_path  = Column(Text,        nullable=True)   # latest checkpoint
    # config
    config_snapshot  = Column(JSON,        nullable=True)   # full resolved config
    config_overrides = Column(JSON,        nullable=True)   # overrides vs experiment base
    # provenance
    parent_run_id    = Column(String(36),  ForeignKey("training_runs.id"), nullable=True)
    resumed_from     = Column(Text,        nullable=True)   # checkpoint path if resumed
    # results
    final_metrics    = Column(JSON,        nullable=True)   # psnr/ssim/lpips at last iter
    best_metrics     = Column(JSON,        nullable=True)   # best per-metric values
    total_iterations = Column(Integer,     nullable=True)
    duration_seconds = Column(Float,       nullable=True)
    # exports
    exports          = Column(JSON,        nullable=True)   # list[ExportRecord]
    # metadata
    hardware_info    = Column(JSON,        nullable=True)   # gpu_name, vram, cuda_version
    tags             = Column(JSON,        nullable=True)
    notes            = Column(Text,        nullable=True)
    created_at       = Column(DateTime,    nullable=False, default=_now)
    started_at       = Column(DateTime,    nullable=True)
    finished_at      = Column(DateTime,    nullable=True)
    updated_at       = Column(DateTime,    nullable=False, default=_now, onupdate=_now)

    # relationships
    project    = relationship("Project",    back_populates="training_runs")
    experiment = relationship("Experiment", back_populates="training_runs")
    dataset    = relationship("Dataset",    back_populates="training_runs")
    parent_run = relationship("TrainingRun", remote_side="TrainingRun.id",
                               foreign_keys=[parent_run_id])
    child_runs = relationship("TrainingRun", back_populates="parent_run",
                               foreign_keys=[parent_run_id])
    metrics    = relationship("RunMetric",  back_populates="training_run",
                               cascade="all, delete-orphan",
                               foreign_keys="RunMetric.training_run_id")
    models     = relationship("Model",      back_populates="training_run",
                               cascade="all, delete-orphan")
    jobs       = relationship("Job",        back_populates="training_run")
    reports    = relationship("Report",     back_populates="training_run")


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class Model(Base):
    """
    A *promoted* checkpoint — a named, versioned artefact that has been
    explicitly saved as a reusable model (as opposed to a transient
    training checkpoint).

    `format`: 'ply' | 'splat' | 'ckpt' | 'onnx' | 'other'
    `stage`:  'draft' | 'staging' | 'production'
    """
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("project_id", "name", "version", name="uq_model_project_name_version"),
    )

    id              = Column(String(36),  primary_key=True, default=_uuid)
    project_id      = Column(String(36),  ForeignKey("projects.id"),       nullable=False)
    training_run_id = Column(String(36),  ForeignKey("training_runs.id"),  nullable=True)
    name            = Column(String(255), nullable=False)
    version         = Column(String(64),  nullable=False, default="1.0.0")
    description     = Column(Text,        nullable=True)
    format          = Column(String(32),  nullable=False, default="ply")
    stage           = Column(String(32),  nullable=False, default="draft")
    file_path       = Column(Text,        nullable=False)
    size_bytes      = Column(BigInteger,  nullable=True)
    checksum_sha256 = Column(String(64),  nullable=True)
    metrics         = Column(JSON,        nullable=True)   # psnr/ssim/lpips snapshot
    metadata_       = Column("metadata",  JSON, nullable=True)
    tags            = Column(JSON,        nullable=True)
    created_at      = Column(DateTime,    nullable=False, default=_now)
    updated_at      = Column(DateTime,    nullable=False, default=_now, onupdate=_now)

    # relationships
    project      = relationship("Project",     back_populates="models")
    training_run = relationship("TrainingRun", back_populates="models")
    reports      = relationship("Report",      back_populates="model")


# ══════════════════════════════════════════════════════════════════════════════
# Reports
# ══════════════════════════════════════════════════════════════════════════════

class Report(Base):
    """
    An evaluation or quality report attached to *any* entity:
    dataset, experiment, training_run, or model.

    Only one of the four FK columns is expected to be non-NULL per row.
    `report_type`: 'dataset_quality' | 'training_eval' | 'model_eval' |
                   'comparison' | 'custom'

    `payload` stores the full structured report dict.
    `file_path` optionally points to a rendered HTML/PDF on disk.
    """
    __tablename__ = "reports"

    id              = Column(String(36),  primary_key=True, default=_uuid)
    project_id      = Column(String(36),  ForeignKey("projects.id"),       nullable=False)
    dataset_id      = Column(String(36),  ForeignKey("datasets.id"),       nullable=True)
    experiment_id   = Column(String(36),  ForeignKey("experiments.id"),    nullable=True)
    training_run_id = Column(String(36),  ForeignKey("training_runs.id"),  nullable=True)
    model_id        = Column(String(36),  ForeignKey("models.id"),         nullable=True)
    report_type     = Column(String(64),  nullable=False, default="custom")
    title           = Column(String(255), nullable=True)
    summary         = Column(Text,        nullable=True)
    payload         = Column(JSON,        nullable=True)   # full report JSON
    file_path       = Column(Text,        nullable=True)   # optional rendered file
    created_at      = Column(DateTime,    nullable=False, default=_now)
    updated_at      = Column(DateTime,    nullable=False, default=_now, onupdate=_now)

    # relationships
    project      = relationship("Project",     back_populates="reports")
    dataset      = relationship("Dataset",     back_populates="reports")
    experiment   = relationship("Experiment",  back_populates="reports")
    training_run = relationship("TrainingRun", back_populates="reports")
    model        = relationship("Model",       back_populates="reports")


# ══════════════════════════════════════════════════════════════════════════════
# Jobs  (async background tasks — Phase 7, extended)
# ══════════════════════════════════════════════════════════════════════════════

class Job(Base):
    """
    One async background task.

    `job_type`: 'upload' | 'analyze' | 'predict' | 'train' | 'export' |
                'evaluate' | 'promote_model'
    `status`:   'pending' | 'running' | 'success' | 'failed' | 'cancelled'
    """
    __tablename__ = "jobs"

    id              = Column(String(36), primary_key=True, default=_uuid)
    project_id      = Column(String(36), ForeignKey("projects.id"),       nullable=True)
    training_run_id = Column(String(36), ForeignKey("training_runs.id"),  nullable=True)
    # legacy FK kept for backwards compat during migration window
    legacy_run_id   = Column(String(36), nullable=True)
    job_type        = Column(String(64), nullable=False)
    status          = Column(String(32), nullable=False, default="pending")
    progress        = Column(Float,      nullable=False, default=0.0)
    message         = Column(Text,       nullable=True)
    result          = Column(JSON,       nullable=True)
    error           = Column(Text,       nullable=True)
    created_at      = Column(DateTime,   nullable=False, default=_now)
    started_at      = Column(DateTime,   nullable=True)
    finished_at     = Column(DateTime,   nullable=True)

    # relationships
    project      = relationship("Project",     back_populates="jobs")
    training_run = relationship("TrainingRun", back_populates="jobs")


# ══════════════════════════════════════════════════════════════════════════════
# RunMetrics  (Phase 7, extended to reference training_runs)
# ══════════════════════════════════════════════════════════════════════════════

class RunMetric(Base):
    """
    Per-iteration scalar metrics for a TrainingRun.

    Both `training_run_id` (Phase 10) and `run_id` (Phase 7 legacy) are
    stored during the migration window; post-migration `run_id` can be
    dropped.
    """
    __tablename__ = "run_metrics"

    id              = Column(Integer,    primary_key=True, autoincrement=True)
    training_run_id = Column(String(36), ForeignKey("training_runs.id"), nullable=True)
    # legacy column — kept until Phase 7 `runs` table is retired
    run_id          = Column(String(36), nullable=True)
    iteration       = Column(Integer,    nullable=False)
    psnr            = Column(Float,      nullable=True)
    ssim            = Column(Float,      nullable=True)
    lpips           = Column(Float,      nullable=True)
    loss            = Column(Float,      nullable=True)
    n_gaussians     = Column(Integer,    nullable=True)
    extra           = Column(JSON,       nullable=True)
    logged_at       = Column(DateTime,   nullable=False, default=_now)

    training_run = relationship("TrainingRun", back_populates="metrics",
                                 foreign_keys=[training_run_id])
