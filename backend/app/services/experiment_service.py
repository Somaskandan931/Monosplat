"""
backend/app/services/experiment_service.py
-------------------------------------------
ExperimentService — CRUD and query helpers for projects, runs, and metrics.
All DB access goes through SQLAlchemy sessions; no raw SQL.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.models.orm import Job, Project, TrainingRun, RunMetric

log = logging.getLogger("monosplat.services.experiment")


class ExperimentService:
    """Read / write operations for projects, runs, and metrics."""

    # ── Projects ──────────────────────────────────────────────────────────────

    def create_project(
        self, db: Session, *, name: str, description: Optional[str] = None
    ) -> Project:
        project = Project(name=name, description=description)
        db.add(project)
        db.commit()
        db.refresh(project)
        log.info("Created project %s (%s)", project.id, project.name)
        return project

    def get_project(self, db: Session, project_id: str) -> Optional[Project]:
        return db.get(Project, project_id)

    def list_projects(self, db: Session, skip: int = 0,
                      limit: int = 100) -> List[Project]:
        return (db.query(Project)
                  .order_by(Project.created_at.desc())
                  .offset(skip).limit(limit).all())

    # ── Runs ──────────────────────────────────────────────────────────────────

    def create_run(
        self,
        db: Session,
        *,
        project_id: Optional[str],
        run_name: str,
        dataset_id: Optional[str] = None,
        sparse_path: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> TrainingRun:
        run = TrainingRun(
            project_id=project_id,
            dataset_id=dataset_id,
            run_name=run_name,
            status="pending",
            sparse_path=sparse_path,
            config_snapshot=config_snapshot,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        log.info("Created run %s (%s)", run.id, run.run_name)
        return run

    def get_run(self, db: Session, run_id: str) -> Optional[TrainingRun]:
        return db.get(TrainingRun, run_id)

    def list_runs(
        self,
        db: Session,
        project_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[TrainingRun]:
        q = db.query(TrainingRun)
        if project_id:
            q = q.filter(TrainingRun.project_id == project_id)
        return q.order_by(TrainingRun.created_at.desc()).offset(skip).limit(limit).all()

    def update_run(
        self,
        db: Session,
        run_id: str,
        *,
        status: Optional[str] = None,
        model_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        finished_at: Optional[datetime] = None,
    ) -> Optional[TrainingRun]:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return None
        if status          is not None: run.status          = status
        if model_path      is not None: run.model_path      = model_path
        if checkpoint_path is not None: run.checkpoint_path = checkpoint_path
        if final_metrics   is not None: run.final_metrics   = final_metrics
        if finished_at     is not None: run.finished_at     = finished_at
        db.commit()
        db.refresh(run)
        return run

    # ── Metrics ───────────────────────────────────────────────────────────────

    def log_metric(
        self,
        db: Session,
        *,
        training_run_id: str,
        iteration: int,
        psnr: Optional[float] = None,
        ssim: Optional[float] = None,
        lpips: Optional[float] = None,
        loss: Optional[float] = None,
        n_gaussians: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RunMetric:
        m = RunMetric(
            training_run_id=training_run_id,
            iteration=iteration,
            psnr=psnr,
            ssim=ssim,
            lpips=lpips,
            loss=loss,
            n_gaussians=n_gaussians,
            extra=extra,
        )
        db.add(m)
        db.commit()
        db.refresh(m)
        return m

    def get_metrics(self, db: Session, training_run_id: str) -> List[RunMetric]:
        return (
            db.query(RunMetric)
            .filter(RunMetric.training_run_id == training_run_id)
            .order_by(RunMetric.iteration)
            .all()
        )

    def get_report(self, db: Session, run_id: str) -> Optional[Dict[str, Any]]:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return None
        return {
            "id":              run.id,
            "project_id":      run.project_id,
            "report_type":     "training_eval",
            "title":           run.run_name,
            "summary":         run.status,
            "payload": {
                "model_path":      run.model_path,
                "config_snapshot": run.config_snapshot,
                "final_metrics":   run.final_metrics,
            },
            "file_path":       None,
            "dataset_id":      run.dataset_id,
            "experiment_id":   run.experiment_id,
            "training_run_id": run.id,
            "model_id":        None,
            "created_at":      run.created_at,
            "updated_at":      run.updated_at,
        }