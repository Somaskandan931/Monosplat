"""
backend/app/schemas/api.py  (Phase 10 — Project & Database Management)
-----------------------------------------------------------------------
Pydantic v2 schemas for all MonoSplat API request / response models.

Naming convention
-----------------
  <Entity>Create  — POST body
  <Entity>Update  — PATCH body (all fields optional)
  <Entity>Read    — response (includes DB-generated IDs / timestamps)
  <Entity>Summary — lightweight read used inside list responses

New in Phase 10
---------------
  User*          — user CRUD
  Dataset*       — dataset lifecycle
  Experiment*    — experiment grouping
  TrainingRun*   — replaces Run* (Phase 7 schemas retained as aliases)
  Model*         — promoted checkpoint management
  Report*        — evaluation/quality reports
  ProjectDetail  — project + nested summaries of all children
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class OKResponse(BaseModel):
    ok: bool = True
    message: str = "success"


class PaginationMeta(BaseModel):
    total: int
    skip: int = 0
    limit: int = 100


# ══════════════════════════════════════════════════════════════════════════════
# Users
# ══════════════════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    username:     str           = Field(..., min_length=3, max_length=128)
    email:        str           = Field(..., max_length=255)
    display_name: Optional[str] = Field(None, max_length=255)
    role:         str           = Field("member", pattern=r"^(admin|member|viewer)$")


class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    role:         Optional[str] = Field(None, pattern=r"^(admin|member|viewer)$")
    is_active:    Optional[bool] = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:           str
    username:     str
    email:        str
    display_name: Optional[str]
    role:         str
    is_active:    bool
    created_at:   datetime
    updated_at:   datetime


# ══════════════════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════════════════

class DatasetCreate(BaseModel):
    project_id:  str
    name:        str           = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    source_type: str           = Field("frames", pattern=r"^(video|frames|zip)$")
    tags:        Optional[List[str]] = None


class DatasetUpdate(BaseModel):
    name:            Optional[str]  = None
    description:     Optional[str]  = None
    status:          Optional[str]  = None
    frames_path:     Optional[str]  = None
    sparse_path:     Optional[str]  = None
    frame_count:     Optional[int]  = None
    resolution_w:    Optional[int]  = None
    resolution_h:    Optional[int]  = None
    size_bytes:      Optional[int]  = None
    analysis_report: Optional[Dict[str, Any]] = None
    quality_score:   Optional[float] = None
    tags:            Optional[List[str]] = None


class DatasetSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:            str
    name:          str
    source_type:   str
    status:        str
    frame_count:   Optional[int]
    quality_score: Optional[float]
    created_at:    datetime


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:              str
    project_id:      str
    name:            str
    description:     Optional[str]
    source_type:     str
    source_path:     Optional[str]
    frames_path:     Optional[str]
    sparse_path:     Optional[str]
    frame_count:     Optional[int]
    resolution_w:    Optional[int]
    resolution_h:    Optional[int]
    size_bytes:      Optional[int]
    status:          str
    analysis_report: Optional[Dict[str, Any]]
    quality_score:   Optional[float]
    tags:            Optional[List[str]]
    created_at:      datetime
    updated_at:      datetime


class DatasetListResponse(BaseModel):
    datasets: List[DatasetSummary]
    meta:     PaginationMeta


# ══════════════════════════════════════════════════════════════════════════════
# Experiments
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentCreate(BaseModel):
    project_id:  str
    name:        str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    base_config: Optional[Dict[str, Any]] = None
    tags:        Optional[List[str]] = None


class ExperimentUpdate(BaseModel):
    name:        Optional[str] = None
    description: Optional[str] = None
    base_config: Optional[Dict[str, Any]] = None
    tags:        Optional[List[str]] = None


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:         str
    name:       str
    created_at: datetime


class ExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:          str
    project_id:  str
    name:        str
    description: Optional[str]
    base_config: Optional[Dict[str, Any]]
    tags:        Optional[List[str]]
    created_at:  datetime
    updated_at:  datetime


class ExperimentListResponse(BaseModel):
    experiments: List[ExperimentSummary]
    meta:        PaginationMeta


# ══════════════════════════════════════════════════════════════════════════════
# Training Runs
# ══════════════════════════════════════════════════════════════════════════════

class TrainingRunCreate(BaseModel):
    project_id:       str
    experiment_id:    Optional[str] = None
    dataset_id:       Optional[str] = None
    run_name:         str = Field(..., min_length=1, max_length=255)
    sparse_path:      str = Field(..., description="Path to COLMAP sparse_text folder")
    image_dir:        str = Field(..., description="Path to extracted frames folder")
    config_overrides: Optional[Dict[str, Any]] = None
    resume_checkpoint: Optional[str] = None
    notes:            Optional[str]  = None
    tags:             Optional[List[str]] = None


class TrainingRunUpdate(BaseModel):
    status:           Optional[str]  = None
    model_path:       Optional[str]  = None
    checkpoint_path:  Optional[str]  = None
    final_metrics:    Optional[Dict[str, Any]] = None
    best_metrics:     Optional[Dict[str, Any]] = None
    total_iterations: Optional[int]  = None
    duration_seconds: Optional[float] = None
    exports:          Optional[List[Dict[str, Any]]] = None
    hardware_info:    Optional[Dict[str, Any]] = None
    notes:            Optional[str]  = None
    tags:             Optional[List[str]] = None
    finished_at:      Optional[datetime] = None


class TrainingRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:            str
    run_name:      str
    status:        str
    experiment_id: Optional[str]
    dataset_id:    Optional[str]
    final_metrics: Optional[Dict[str, Any]]
    created_at:    datetime
    finished_at:   Optional[datetime]


class TrainingRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:               str
    project_id:       Optional[str]
    experiment_id:    Optional[str]
    dataset_id:       Optional[str]
    run_name:         str
    status:           str
    sparse_path:      Optional[str]
    model_path:       Optional[str]
    checkpoint_path:  Optional[str]
    config_snapshot:  Optional[Dict[str, Any]]
    config_overrides: Optional[Dict[str, Any]]
    parent_run_id:    Optional[str]
    resumed_from:     Optional[str]
    final_metrics:    Optional[Dict[str, Any]]
    best_metrics:     Optional[Dict[str, Any]]
    total_iterations: Optional[int]
    duration_seconds: Optional[float]
    exports:          Optional[List[Dict[str, Any]]]
    hardware_info:    Optional[Dict[str, Any]]
    tags:             Optional[List[str]]
    notes:            Optional[str]
    created_at:       datetime
    started_at:       Optional[datetime]
    finished_at:      Optional[datetime]
    updated_at:       datetime


class TrainingRunListResponse(BaseModel):
    runs: List[TrainingRunSummary]
    meta: PaginationMeta


# ── Phase 7 backward-compat aliases ───────────────────────────────────────────
RunRead          = TrainingRunSummary
RunsListResponse = TrainingRunListResponse


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class ModelCreate(BaseModel):
    project_id:      str
    training_run_id: Optional[str]  = None
    name:            str = Field(..., min_length=1, max_length=255)
    version:         str = Field("1.0.0", max_length=64)
    description:     Optional[str]  = None
    format:          str = Field("ply", pattern=r"^(ply|splat|ckpt|onnx|other)$")
    stage:           str = Field("draft", pattern=r"^(draft|staging|production)$")
    file_path:       str
    size_bytes:      Optional[int]  = None
    checksum_sha256: Optional[str]  = None
    metrics:         Optional[Dict[str, Any]] = None
    tags:            Optional[List[str]] = None


class ModelUpdate(BaseModel):
    description:     Optional[str]  = None
    stage:           Optional[str]  = Field(None, pattern=r"^(draft|staging|production)$")
    metrics:         Optional[Dict[str, Any]] = None
    checksum_sha256: Optional[str]  = None
    tags:            Optional[List[str]] = None


class ModelSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:         str
    name:       str
    version:    str
    format:     str
    stage:      str
    created_at: datetime


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:              str
    project_id:      str
    training_run_id: Optional[str]
    name:            str
    version:         str
    description:     Optional[str]
    format:          str
    stage:           str
    file_path:       str
    size_bytes:      Optional[int]
    checksum_sha256: Optional[str]
    metrics:         Optional[Dict[str, Any]]
    tags:            Optional[List[str]]
    created_at:      datetime
    updated_at:      datetime


class ModelListResponse(BaseModel):
    models: List[ModelSummary]
    meta:   PaginationMeta


# ══════════════════════════════════════════════════════════════════════════════
# Reports
# ══════════════════════════════════════════════════════════════════════════════

class ReportCreate(BaseModel):
    project_id:      str
    report_type:     str = Field("custom",
                        pattern=r"^(dataset_quality|training_eval|model_eval|comparison|custom)$")
    title:           Optional[str]  = None
    summary:         Optional[str]  = None
    payload:         Optional[Dict[str, Any]] = None
    file_path:       Optional[str]  = None
    # link to at most one entity
    dataset_id:      Optional[str]  = None
    experiment_id:   Optional[str]  = None
    training_run_id: Optional[str]  = None
    model_id:        Optional[str]  = None


class ReportUpdate(BaseModel):
    title:    Optional[str]  = None
    summary:  Optional[str]  = None
    payload:  Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None


class ReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:              str
    project_id:      str
    report_type:     str
    title:           Optional[str]
    summary:         Optional[str]
    payload:         Optional[Dict[str, Any]]
    file_path:       Optional[str]
    dataset_id:      Optional[str]
    experiment_id:   Optional[str]
    training_run_id: Optional[str]
    model_id:        Optional[str]
    created_at:      datetime
    updated_at:      datetime


class ReportListResponse(BaseModel):
    reports: List[ReportRead]
    meta:    PaginationMeta


# ══════════════════════════════════════════════════════════════════════════════
# Projects  (extended from Phase 7)
# ══════════════════════════════════════════════════════════════════════════════

class ProjectCreate(BaseModel):
    name:        str = Field(..., min_length=1, max_length=255)
    description: Optional[str]  = None
    owner_id:    Optional[str]  = None
    tags:        Optional[List[str]] = None
    settings:    Optional[Dict[str, Any]] = None


class ProjectUpdate(BaseModel):
    name:                 Optional[str]  = None
    description:          Optional[str]  = None
    tags:                 Optional[List[str]] = None
    settings:             Optional[Dict[str, Any]] = None
    active_dataset_id:    Optional[str]  = None
    active_experiment_id: Optional[str]  = None


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:                   str
    owner_id:             Optional[str]
    name:                 str
    description:          Optional[str]
    tags:                 Optional[List[str]]
    settings:             Optional[Dict[str, Any]]
    active_dataset_id:    Optional[str]
    active_experiment_id: Optional[str]
    created_at:           datetime
    updated_at:           datetime


class ProjectDetail(ProjectRead):
    """Full project read with nested summaries of all child entities."""
    datasets:      List[DatasetSummary]     = []
    experiments:   List[ExperimentSummary]  = []
    training_runs: List[TrainingRunSummary] = []
    models:        List[ModelSummary]       = []
    reports:       List[ReportRead]         = []


class ProjectsListResponse(BaseModel):
    projects: List[ProjectRead]
    meta:     PaginationMeta


# ══════════════════════════════════════════════════════════════════════════════
# Jobs  (Phase 7, lightly extended)
# ══════════════════════════════════════════════════════════════════════════════

class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:              str
    project_id:      Optional[str]
    training_run_id: Optional[str]
    job_type:        str
    status:          str
    progress:        float
    message:         Optional[str]
    result:          Optional[Dict[str, Any]]
    error:           Optional[str]
    created_at:      datetime
    started_at:      Optional[datetime]
    finished_at:     Optional[datetime]


StatusResponse = JobRead


# ══════════════════════════════════════════════════════════════════════════════
# Metrics  (Phase 7 unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class MetricPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    iteration:   int
    psnr:        Optional[float]
    ssim:        Optional[float]
    lpips:       Optional[float]
    loss:        Optional[float]
    n_gaussians: Optional[int]
    extra:       Optional[Dict[str, Any]]
    logged_at:   datetime


class MetricsResponse(BaseModel):
    run_id:  str
    metrics: List[MetricPoint]
    count:   int


# ══════════════════════════════════════════════════════════════════════════════
# Upload / Analyze / Predict / Train / Export  (Phase 7 kept, extended)
# ══════════════════════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    job_id:      str
    project_id:  str
    dataset_id:  Optional[str]
    upload_path: str
    frames_path: Optional[str] = None   # directory of extracted frames; None until extraction completes
    filename:    str
    size_bytes:  int
    message:     str = "upload queued"


class AnalyzeRequest(BaseModel):
    project_id:     str
    dataset_id:     Optional[str] = None
    image_dir:      str = Field(..., description="Server-side path to extracted frames")
    blur_threshold: float = Field(120.0, ge=0.0)


class AnalyzeResponse(BaseModel):
    job_id:     str
    project_id: str
    dataset_id: Optional[str]
    image_dir:  str
    message:    str = "analysis queued"


class PredictRequest(BaseModel):
    project_id:      str
    analysis_report: Dict[str, Any]


class PredictResponse(BaseModel):
    job_id:     str
    project_id: str
    message:    str = "prediction queued"


class TrainRequest(BaseModel):
    project_id:        str
    experiment_id:     Optional[str] = None
    dataset_id:        Optional[str] = None
    sparse_path:       str
    image_dir:         str
    config_overrides:  Optional[Dict[str, Any]] = None
    resume_checkpoint: Optional[str] = None
    notes:             Optional[str] = None
    tags:              Optional[List[str]] = None


class TrainResponse(BaseModel):
    job_id:     str
    run_id:     str
    project_id: str
    message:    str = "training job queued"


class ResumeRequest(BaseModel):
    run_id:           str
    checkpoint_path:  Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None


class ResumeResponse(BaseModel):
    job_id:          str
    run_id:          str
    checkpoint_path: str
    message:         str = "resume job queued"


class ExportRequest(BaseModel):
    run_id:     str
    formats:    List[str] = Field(default=["ply", "splat"])
    output_dir: Optional[str] = None


class ExportResponse(BaseModel):
    job_id:  str
    run_id:  str
    formats: List[str]
    message: str = "export job queued"


# ══════════════════════════════════════════════════════════════════════════════
# Model promotion
# ══════════════════════════════════════════════════════════════════════════════

class PromoteModelRequest(BaseModel):
    """Promote a training run checkpoint to a named Model artefact."""
    run_id:      str
    name:        str = Field(..., min_length=1, max_length=255)
    version:     str = Field("1.0.0")
    format:      str = Field("ply", pattern=r"^(ply|splat|ckpt|onnx|other)$")
    stage:       str = Field("draft", pattern=r"^(draft|staging|production)$")
    description: Optional[str] = None
    tags:        Optional[List[str]] = None


class PromoteModelResponse(BaseModel):
    job_id:   str
    model_id: str
    run_id:   str
    message:  str = "model promotion job queued"