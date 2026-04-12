"""
pipeline_manager.py
Dynamic pipeline manager — watches an upload inbox and automatically:
    1. Extracts frames (FFmpeg)        →  work/<job_id>/frames/
    2. Runs COLMAP pose estimation     →  work/<job_id>/colmap/
    3. (Optional) Trains Gaussian Splat →  work/<job_id>/models/  (GPU only)
    4. Exports .ply + .splat
    5. Registers the model in registry →  models/registry.json

Architecture
------------
- PipelineWorker._dispatch() delegates entirely to worker.run_pipeline().
  All pipeline stage logic lives in one place (worker.py) — no duplication.
- ModelRegistry uses atomic write (tmp → rename) to prevent corruption.
- UploadWatcher requires STABILITY_WAIT seconds of no mtime change before firing.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Status & data classes
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    WAITING    = "waiting"
    EXTRACTING = "extracting"
    COLMAP     = "colmap"
    TRAINING   = "training"
    EXPORTING  = "exporting"
    READY      = "ready"
    READY_FOR_COLAB = "ready_for_colab"  # COLMAP done, ready for GPU training in Colab
    FAILED     = "failed"


@dataclass
class ModelJob:
    job_id:        str
    item_name:     str
    upload_dir:    str
    work_dir:      str
    status:        JobStatus = JobStatus.WAITING
    progress:      int       = 0
    message:       str       = ""
    ply_path:      Optional[str] = None
    splat_path:    Optional[str] = None
    thumbnail:     Optional[str] = None
    num_images:    int       = 0
    num_gaussians: int       = 0
    error:         Optional[str] = None
    created_at:    str       = field(default_factory=lambda: datetime.now().isoformat())
    updated_at:    str       = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def update(self, status: JobStatus = None, progress: int = None,
               message: str = None, **kwargs):
        if status is not None:
            if isinstance(status, str):
                try:
                    status = JobStatus(status)
                except ValueError:
                    status = JobStatus.FAILED
            self.status = status
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.updated_at = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Thread-safe JSON registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    JSON-backed registry of all model jobs.
    Uses atomic write (tmp → rename) to prevent file corruption on crash.
    """

    def __init__(self, registry_path: str = "models/registry.json"):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock: threading.Lock = threading.Lock()
        self._jobs: Dict[str, ModelJob] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                raw = json.load(f)
            for jid, jdata in raw.items():
                try:
                    jdata["status"] = JobStatus(jdata["status"])
                except ValueError:
                    jdata["status"] = JobStatus.FAILED
                self._jobs[jid] = ModelJob(**jdata)
        except Exception as e:
            print(f"[registry] Failed to load registry: {e}  (starting fresh)")

    def _save(self):
        import os, shutil
        data = {jid: job.to_dict() for jid, job in self._jobs.items()}
        tmp  = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        # Windows does not allow rename over an open/locked file the way POSIX does.
        # Fall back to remove-then-move if the atomic replace raises PermissionError.
        try:
            tmp.replace(self.path)
        except PermissionError:
            if self.path.exists():
                os.remove(self.path)
            shutil.move(str(tmp), str(self.path))

    def add_job(self, job: ModelJob):
        with self._lock:
            self._jobs[job.job_id] = job
            self._save()

    def update_job(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(**kwargs)
                self._save()

    def get_job(self, job_id: str) -> Optional[ModelJob]:
        return self._jobs.get(job_id)

    def all_jobs(self) -> List[ModelJob]:
        return list(self._jobs.values())

    def ready_models(self) -> List[ModelJob]:
        return [j for j in self._jobs.values() if j.status == JobStatus.READY]

    def job_exists(self, item_name: str) -> bool:
        """True if an actively-running job exists for this item_name.
        WAITING is treated as orphaned so users can re-submit after a restart."""
        active = {JobStatus.EXTRACTING, JobStatus.COLMAP,
                  JobStatus.TRAINING, JobStatus.EXPORTING, JobStatus.READY_FOR_COLAB}
        return any(
            j.item_name == item_name and j.status in active
            for j in self._jobs.values()
        )

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            _ACTIVE = {JobStatus.EXTRACTING, JobStatus.COLMAP,
                       JobStatus.TRAINING, JobStatus.EXPORTING, JobStatus.READY_FOR_COLAB}
            if job.status in _ACTIVE:
                return False
            del self._jobs[job_id]
            self._save()
            return True

    def clear_all_inactive(self) -> int:
        with self._lock:
            _ACTIVE = {JobStatus.EXTRACTING, JobStatus.COLMAP,
                       JobStatus.TRAINING, JobStatus.EXPORTING, JobStatus.READY_FOR_COLAB}
            ids = [jid for jid, j in self._jobs.items() if j.status not in _ACTIVE]
            for jid in ids:
                del self._jobs[jid]
            if ids:
                self._save()
            return len(ids)

    def clear_failed(self) -> int:
        with self._lock:
            ids = [jid for jid, j in self._jobs.items()
                   if j.status in (JobStatus.FAILED, JobStatus.WAITING)]
            for jid in ids:
                del self._jobs[jid]
            if ids:
                self._save()
            return len(ids)

    def to_json(self) -> str:
        return json.dumps([j.to_dict() for j in self._jobs.values()], indent=2)


# ---------------------------------------------------------------------------
# Upload watcher
# ---------------------------------------------------------------------------

class UploadWatcher:
    """
    Polls uploads/ for new subdirectories and triggers a pipeline job once the
    directory has been stable (no mtime change) for STABILITY_WAIT seconds.
    """

    STABILITY_WAIT = 5.0

    def __init__(self, inbox: str, on_new_upload, poll_interval: float = 2.0):
        self.inbox         = Path(inbox)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.on_new_upload = on_new_upload
        self.poll_interval = poll_interval
        self._seen:      Dict[str, float] = {}
        self._stable:    Dict[str, float] = {}
        self._processed: set              = set()
        self._stop_event = threading.Event()

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True, name="UploadWatcher")
        t.start()
        print(f"[watcher] Watching: {self.inbox.resolve()}")

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.wait(timeout=self.poll_interval):
            self._scan()

    def _scan(self):
        now = time.monotonic()
        try:
            entries = sorted(self.inbox.iterdir())
        except OSError:
            return

        for item_dir in entries:
            if not item_dir.is_dir():
                continue
            key = item_dir.name
            if key in self._processed:
                continue

            mtimes = [f.stat().st_mtime for f in item_dir.rglob("*") if f.is_file()]
            if not mtimes:
                continue
            last_mtime = max(mtimes)

            if self._seen.get(key) != last_mtime:
                self._seen[key]   = last_mtime
                self._stable[key] = now
            elif now - self._stable.get(key, now) >= self.STABILITY_WAIT:
                print(f"[watcher] Stable upload detected: {key}")
                job_id = self.on_new_upload(str(item_dir), key)
                if job_id:
                    self._processed.add(key)


# ---------------------------------------------------------------------------
# Pipeline worker
# ---------------------------------------------------------------------------

class PipelineWorker:
    """
    Manages the job queue and background thread.
    All pipeline logic is delegated to worker.run_pipeline() — no duplication.
    """

    def __init__(self, registry: ModelRegistry, cfg, work_base: str = "work"):
        self.registry    = registry
        self.cfg         = cfg
        self.work_base   = Path(work_base)
        self._queue:     List[ModelJob] = []
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()

    def enqueue(self, job: ModelJob):
        with self._lock:
            self._queue.append(job)
        print(f"[worker] Queued: {job.item_name} ({job.job_id})")

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True, name="PipelineWorker")
        t.start()
        # Recover WAITING jobs orphaned by a previous restart
        for job in self.registry.all_jobs():
            if job.status == JobStatus.WAITING:
                self._queue.append(job)
                print(f"[worker] Recovered orphaned job: {job.item_name} ({job.job_id})")

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            job = None
            with self._lock:
                if self._queue:
                    job = self._queue.pop(0)
            if job:
                self._dispatch(job)
            else:
                self._stop_event.wait(timeout=1.0)

    def _dispatch(self, job: ModelJob):
        """Delegate entirely to worker.run_pipeline() — single source of truth."""
        print(f"\n[worker] ▶  {job.item_name}  ({job.job_id})")

        work_dir = self.work_base / job.job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        upload_path = Path(job.upload_dir)
        video_exts  = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"}
        videos      = [p for p in upload_path.iterdir() if p.suffix in video_exts]
        input_path  = str(videos[0]) if videos else str(upload_path)

        log_dir  = Path("outputs/logs")
        log_file = open(log_dir / f"{job.job_id}.log", "w", encoding="utf-8") \
                   if log_dir.exists() else None

        def on_progress(step: str, line: str):
            msg = f"[COLMAP] {step}: {line[:80]}"
            self.registry.update_job(job.job_id, message=msg)
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()

        try:
            from src.pipeline.worker import run_pipeline
            run_pipeline(
                job_id=job.job_id,
                input_path=input_path,
                work_dir=str(work_dir),
                config_path="config/config.yaml",
                registry=self.registry,
                on_progress=on_progress,
            )
        finally:
            if log_file:
                log_file.close()


# ---------------------------------------------------------------------------
# Top-level manager
# ---------------------------------------------------------------------------

class DynamicPipelineManager:
    """
    Top-level manager. Wire up at app startup via server.py.

    Usage:
        manager = DynamicPipelineManager()
        manager.start()
        # Drop folders into uploads/ — everything is automatic.
    """

    def __init__(
        self,
        inbox:         str = "uploads",
        registry_path: str = "models/registry.json",
        work_base:     str = "work",
        config_path:   str = "config/config.yaml",
    ):
        from src.utils.config_loader import load_config
        self.cfg      = load_config(config_path)
        self.registry = ModelRegistry(registry_path)
        self.worker   = PipelineWorker(self.registry, self.cfg, work_base)
        self.watcher  = UploadWatcher(inbox, self._on_new_upload)

    def _make_job_id(self, item_name: str) -> str:
        raw = f"{item_name}{time.monotonic()}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _on_new_upload(self, upload_dir: str, item_name: str) -> str:
        if self.registry.job_exists(item_name):
            print(f"[manager] Skipping duplicate job for '{item_name}'")
            return ""
        job_id   = self._make_job_id(item_name)
        work_dir = str(Path("work") / job_id)
        job = ModelJob(
            job_id=job_id,
            item_name=item_name,
            upload_dir=upload_dir,
            work_dir=work_dir,
        )
        self.registry.add_job(job)
        self.worker.enqueue(job)
        print(f"[manager] New job: '{item_name}'  →  {job_id}")
        return job_id

    def submit_job(self, upload_dir: str, item_name: str) -> str:
        return self._on_new_upload(upload_dir, item_name)

    def start(self):
        self.worker.start()
        self.watcher.start()
        print("[manager] Dynamic pipeline started.")
        print("[manager] Drop folders into 'uploads/' to auto-process.")

    def stop(self):
        self.worker.stop()
        self.watcher.stop()
        print("[manager] Stopped.")

    def get_registry(self) -> ModelRegistry:
        return self.registry