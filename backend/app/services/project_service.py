"""
backend/app/services/project_service.py  (Phase 10)
-----------------------------------------------------
CRUD + query helpers for the full Phase 10 schema.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.models.orm import (
    Dataset,
    Experiment,
    Job,
    Model,
    Project,
    Report,
    RunMetric,
    TrainingRun,
    User,
)

log = logging.getLogger("monosplat.services.project")


def _apply(obj, **kwargs) -> None:
    for k, v in kwargs.items():
        if v is not None:
            setattr(obj, k, v)


def _hash_api_key(raw_key: str, salt: str = "monosplat") -> str:
    return hashlib.sha256(f"{salt}:{raw_key}".encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Users
# ══════════════════════════════════════════════════════════════════════════════

class UserService:

    def create(self, db: Session, *, username: str, email: str,
               display_name: Optional[str] = None,
               role: str = "member") -> User:
        user = User(username=username, email=email,
                    display_name=display_name, role=role)
        db.add(user)
        db.flush()
        log.info("Created user %s (%s)", user.id, username)
        return user

    def get(self, db: Session, user_id: str) -> Optional[User]:
        return db.get(User, user_id)

    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()

    def get_by_api_key(self, db: Session, raw_key: str) -> Optional[User]:
        h = _hash_api_key(raw_key)
        return db.query(User).filter(User.api_key_hash == h,
                                      User.is_active == True).first()  # noqa: E712

    def list(self, db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        return (db.query(User)
                  .order_by(User.created_at.desc())
                  .offset(skip).limit(limit).all())

    def update(self, db: Session, user_id: str, **kwargs) -> Optional[User]:
        user = db.get(User, user_id)
        if user is None:
            return None
        _apply(user, **kwargs)
        db.flush()
        return user

    def set_api_key(self, db: Session, user_id: str, raw_key: str) -> Optional[User]:
        user = db.get(User, user_id)
        if user is None:
            return None
        user.api_key_hash = _hash_api_key(raw_key)
        db.flush()
        return user

    def delete(self, db: Session, user_id: str) -> bool:
        user = db.get(User, user_id)
        if user is None:
            return False
        db.delete(user)
        db.flush()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Projects
# ══════════════════════════════════════════════════════════════════════════════

class ProjectService:

    def create(self, db: Session, *, name: str,
               description: Optional[str] = None,
               owner_id: Optional[str] = None,
               tags: Optional[List[str]] = None,
               settings: Optional[Dict[str, Any]] = None) -> Project:
        project = Project(name=name, description=description,
                           owner_id=owner_id, tags=tags, settings=settings)
        db.add(project)
        db.flush()
        log.info("Created project %s (%s)", project.id, name)
        return project

    def get(self, db: Session, project_id: str) -> Optional[Project]:
        return db.get(Project, project_id)

    def list(self, db: Session, owner_id: Optional[str] = None,
             skip: int = 0, limit: int = 100) -> List[Project]:
        q = db.query(Project)
        if owner_id:
            q = q.filter(Project.owner_id == owner_id)
        return q.order_by(Project.created_at.desc()).offset(skip).limit(limit).all()

    def update(self, db: Session, project_id: str, **kwargs) -> Optional[Project]:
        project = db.get(Project, project_id)
        if project is None:
            return None
        _apply(project, **kwargs)
        db.flush()
        return project

    def set_active_dataset(self, db: Session, project_id: str,
                            dataset_id: str) -> Optional[Project]:
        return self.update(db, project_id, active_dataset_id=dataset_id)

    def set_active_experiment(self, db: Session, project_id: str,
                               experiment_id: str) -> Optional[Project]:
        return self.update(db, project_id, active_experiment_id=experiment_id)

    def delete(self, db: Session, project_id: str) -> bool:
        project = db.get(Project, project_id)
        if project is None:
            return False
        db.delete(project)
        db.flush()
        return True

    def get_detail(self, db: Session, project_id: str) -> Optional[Project]:
        from sqlalchemy.orm import joinedload
        return (db.query(Project)
                  .options(
                      joinedload(Project.datasets),
                      joinedload(Project.experiments),
                      joinedload(Project.training_runs),
                      joinedload(Project.models),
                      joinedload(Project.reports),
                  )
                  .filter(Project.id == project_id)
                  .first())


# ══════════════════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════════════════

class DatasetService:

    def create(self, db: Session, *, project_id: str, name: str,
               source_type: str = "frames",
               description: Optional[str] = None,
               source_path: Optional[str] = None,
               tags: Optional[List[str]] = None) -> Dataset:
        ds = Dataset(project_id=project_id, name=name,
                      source_type=source_type, description=description,
                      source_path=source_path, tags=tags, status="uploading")
        db.add(ds)
        db.flush()
        log.info("Created dataset %s (%s) in project %s", ds.id, name, project_id)
        return ds

    def get(self, db: Session, dataset_id: str) -> Optional[Dataset]:
        return db.get(Dataset, dataset_id)

    def list(self, db: Session, project_id: str,
             skip: int = 0, limit: int = 100) -> List[Dataset]:
        return (db.query(Dataset)
                  .filter(Dataset.project_id == project_id)
                  .order_by(Dataset.created_at.desc())
                  .offset(skip).limit(limit).all())

    def update(self, db: Session, dataset_id: str, **kwargs) -> Optional[Dataset]:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            return None
        _apply(ds, **kwargs)
        db.flush()
        return ds

    def mark_ready(self, db: Session, dataset_id: str,
                   frames_path: str, sparse_path: Optional[str] = None,
                   frame_count: Optional[int] = None,
                   resolution_w: Optional[int] = None,
                   resolution_h: Optional[int] = None) -> Optional[Dataset]:
        return self.update(db, dataset_id,
                           status="ready",
                           frames_path=frames_path,
                           sparse_path=sparse_path,
                           frame_count=frame_count,
                           resolution_w=resolution_w,
                           resolution_h=resolution_h)

    def store_analysis(self, db: Session, dataset_id: str,
                       report: Dict[str, Any],
                       quality_score: Optional[float] = None) -> Optional[Dataset]:
        return self.update(db, dataset_id,
                           status="analyzed",
                           analysis_report=report,
                           quality_score=quality_score)

    def delete(self, db: Session, dataset_id: str) -> bool:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            return False
        db.delete(ds)
        db.flush()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Experiments
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentService:

    def create(self, db: Session, *, project_id: str, name: str,
               description: Optional[str] = None,
               base_config: Optional[Dict[str, Any]] = None,
               tags: Optional[List[str]] = None) -> Experiment:
        exp = Experiment(project_id=project_id, name=name,
                          description=description, base_config=base_config,
                          tags=tags)
        db.add(exp)
        db.flush()
        log.info("Created experiment %s (%s) in project %s", exp.id, name, project_id)
        return exp

    def get(self, db: Session, experiment_id: str) -> Optional[Experiment]:
        return db.get(Experiment, experiment_id)

    def list(self, db: Session, project_id: str,
             skip: int = 0, limit: int = 100) -> List[Experiment]:
        return (db.query(Experiment)
                  .filter(Experiment.project_id == project_id)
                  .order_by(Experiment.created_at.desc())
                  .offset(skip).limit(limit).all())

    def update(self, db: Session, experiment_id: str,
               **kwargs) -> Optional[Experiment]:
        exp = db.get(Experiment, experiment_id)
        if exp is None:
            return None
        _apply(exp, **kwargs)
        db.flush()
        return exp

    def delete(self, db: Session, experiment_id: str) -> bool:
        exp = db.get(Experiment, experiment_id)
        if exp is None:
            return False
        db.delete(exp)
        db.flush()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Training Runs
# ══════════════════════════════════════════════════════════════════════════════

class TrainingRunService:

    def create(self, db: Session, *, project_id: Optional[str],
               run_name: str,
               experiment_id: Optional[str] = None,
               dataset_id: Optional[str] = None,
               sparse_path: Optional[str] = None,
               config_snapshot: Optional[Dict[str, Any]] = None,
               config_overrides: Optional[Dict[str, Any]] = None,
               notes: Optional[str] = None,
               tags: Optional[List[str]] = None) -> TrainingRun:
        run = TrainingRun(
            project_id=project_id,
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            run_name=run_name,
            status="pending",
            sparse_path=sparse_path,
            config_snapshot=config_snapshot,
            config_overrides=config_overrides,
            notes=notes,
            tags=tags,
        )
        db.add(run)
        db.flush()
        log.info("Created training_run %s (%s)", run.id, run_name)
        return run

    def get(self, db: Session, run_id: str) -> Optional[TrainingRun]:
        return db.get(TrainingRun, run_id)

    def list(self, db: Session,
             project_id: Optional[str] = None,
             experiment_id: Optional[str] = None,
             dataset_id: Optional[str] = None,
             status: Optional[str] = None,
             skip: int = 0, limit: int = 100) -> List[TrainingRun]:
        q = db.query(TrainingRun)
        if project_id:
            q = q.filter(TrainingRun.project_id == project_id)
        if experiment_id:
            q = q.filter(TrainingRun.experiment_id == experiment_id)
        if dataset_id:
            q = q.filter(TrainingRun.dataset_id == dataset_id)
        if status:
            q = q.filter(TrainingRun.status == status)
        return q.order_by(TrainingRun.created_at.desc()).offset(skip).limit(limit).all()

    def update(self, db: Session, run_id: str, **kwargs) -> Optional[TrainingRun]:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return None
        _apply(run, **kwargs)
        db.flush()
        return run

    def mark_started(self, db: Session, run_id: str) -> Optional[TrainingRun]:
        return self.update(db, run_id,
                           status="running",
                           started_at=datetime.utcnow())

    def mark_finished(self, db: Session, run_id: str,
                      success: bool,
                      model_path: Optional[str] = None,
                      checkpoint_path: Optional[str] = None,
                      final_metrics: Optional[Dict[str, Any]] = None,
                      best_metrics: Optional[Dict[str, Any]] = None,
                      total_iterations: Optional[int] = None,
                      duration_seconds: Optional[float] = None,
                      hardware_info: Optional[Dict[str, Any]] = None) -> Optional[TrainingRun]:
        return self.update(db, run_id,
                           status="success" if success else "failed",
                           model_path=model_path,
                           checkpoint_path=checkpoint_path,
                           final_metrics=final_metrics,
                           best_metrics=best_metrics,
                           total_iterations=total_iterations,
                           duration_seconds=duration_seconds,
                           hardware_info=hardware_info,
                           finished_at=datetime.utcnow())

    def add_export(self, db: Session, run_id: str,
                   export_record: Dict[str, Any]) -> Optional[TrainingRun]:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return None
        existing = list(run.exports or [])
        existing.append(export_record)
        run.exports = existing
        db.flush()
        return run

    def log_metric(self, db: Session, *, training_run_id: str, iteration: int,
                   psnr: Optional[float] = None,
                   ssim: Optional[float] = None,
                   lpips: Optional[float] = None,
                   loss: Optional[float] = None,
                   n_gaussians: Optional[int] = None,
                   extra: Optional[Dict[str, Any]] = None) -> RunMetric:
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
        db.flush()
        return m

    def get_metrics(self, db: Session, training_run_id: str) -> List[RunMetric]:
        return (db.query(RunMetric)
                  .filter(RunMetric.training_run_id == training_run_id)
                  .order_by(RunMetric.iteration)
                  .all())

    def delete(self, db: Session, run_id: str) -> bool:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return False
        db.delete(run)
        db.flush()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class ModelService:

    def create(self, db: Session, *, project_id: str, name: str,
               file_path: str,
               training_run_id: Optional[str] = None,
               version: str = "1.0.0",
               format: str = "ply",
               stage: str = "draft",
               description: Optional[str] = None,
               size_bytes: Optional[int] = None,
               checksum_sha256: Optional[str] = None,
               metrics: Optional[Dict[str, Any]] = None,
               tags: Optional[List[str]] = None) -> Model:
        model = Model(
            project_id=project_id,
            training_run_id=training_run_id,
            name=name,
            version=version,
            format=format,
            stage=stage,
            file_path=file_path,
            description=description,
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
            metrics=metrics,
            tags=tags,
        )
        db.add(model)
        db.flush()
        log.info("Created model %s v%s (%s) in project %s",
                 name, version, model.id, project_id)
        return model

    def get(self, db: Session, model_id: str) -> Optional[Model]:
        return db.get(Model, model_id)

    def list(self, db: Session, project_id: str,
             stage: Optional[str] = None,
             format: Optional[str] = None,
             skip: int = 0, limit: int = 100) -> List[Model]:
        q = db.query(Model).filter(Model.project_id == project_id)
        if stage:
            q = q.filter(Model.stage == stage)
        if format:
            q = q.filter(Model.format == format)
        return q.order_by(Model.created_at.desc()).offset(skip).limit(limit).all()

    def update(self, db: Session, model_id: str, **kwargs) -> Optional[Model]:
        model = db.get(Model, model_id)
        if model is None:
            return None
        _apply(model, **kwargs)
        db.flush()
        return model

    def promote(self, db: Session, model_id: str, stage: str) -> Optional[Model]:
        return self.update(db, model_id, stage=stage)

    def delete(self, db: Session, model_id: str) -> bool:
        model = db.get(Model, model_id)
        if model is None:
            return False
        db.delete(model)
        db.flush()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Reports
# ══════════════════════════════════════════════════════════════════════════════

class ReportService:

    def create(self, db: Session, *, project_id: str,
               report_type: str = "custom",
               title: Optional[str] = None,
               summary: Optional[str] = None,
               payload: Optional[Dict[str, Any]] = None,
               file_path: Optional[str] = None,
               dataset_id: Optional[str] = None,
               experiment_id: Optional[str] = None,
               training_run_id: Optional[str] = None,
               model_id: Optional[str] = None) -> Report:
        report = Report(
            project_id=project_id,
            report_type=report_type,
            title=title,
            summary=summary,
            payload=payload,
            file_path=file_path,
            dataset_id=dataset_id,
            experiment_id=experiment_id,
            training_run_id=training_run_id,
            model_id=model_id,
        )
        db.add(report)
        db.flush()
        log.info("Created report %s (type=%s) in project %s",
                 report.id, report_type, project_id)
        return report

    def get(self, db: Session, report_id: str) -> Optional[Report]:
        return db.get(Report, report_id)

    def list(self, db: Session, project_id: str,
             report_type: Optional[str] = None,
             training_run_id: Optional[str] = None,
             skip: int = 0, limit: int = 100) -> List[Report]:
        q = db.query(Report).filter(Report.project_id == project_id)
        if report_type:
            q = q.filter(Report.report_type == report_type)
        if training_run_id:
            q = q.filter(Report.training_run_id == training_run_id)
        return q.order_by(Report.created_at.desc()).offset(skip).limit(limit).all()

    def update(self, db: Session, report_id: str, **kwargs) -> Optional[Report]:
        report = db.get(Report, report_id)
        if report is None:
            return None
        _apply(report, **kwargs)
        db.flush()
        return report

    def delete(self, db: Session, report_id: str) -> bool:
        report = db.get(Report, report_id)
        if report is None:
            return False
        db.delete(report)
        db.flush()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Jobs
# ══════════════════════════════════════════════════════════════════════════════

class JobService:

    def create(self, db: Session, *, job_type: str,
               project_id: Optional[str] = None,
               training_run_id: Optional[str] = None) -> Job:
        job = Job(job_type=job_type,
                   project_id=project_id,
                   training_run_id=training_run_id)
        db.add(job)
        db.flush()
        return job

    def get(self, db: Session, job_id: str) -> Optional[Job]:
        return db.get(Job, job_id)

    def list(self, db: Session,
             project_id: Optional[str] = None,
             status: Optional[str] = None,
             skip: int = 0, limit: int = 100) -> List[Job]:
        q = db.query(Job)
        if project_id:
            q = q.filter(Job.project_id == project_id)
        if status:
            q = q.filter(Job.status == status)
        return q.order_by(Job.created_at.desc()).offset(skip).limit(limit).all()

    def update(self, db: Session, job_id: str, **kwargs) -> Optional[Job]:
        job = db.get(Job, job_id)
        if job is None:
            return None
        _apply(job, **kwargs)
        db.flush()
        return job