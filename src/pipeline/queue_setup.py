"""
queue_setup.py
Redis + RQ job queue setup for MonoSplat.

Provides a single shared Queue instance used by:
    - server.py    → to enqueue jobs
    - worker.py    → consumed by the RQ worker process

Usage
-----
    from src.pipeline.queue_setup import get_queue, enqueue_pipeline

Run the worker (separate terminal):
    rq worker monosplat --url redis://localhost:6379

If Redis is not available, falls back to a lightweight in-process
thread-based queue so the system still works without Redis installed.
"""

import threading
from queue import Queue as ThreadQueue
from typing import Optional

# ---------------------------------------------------------------------------
# Redis / RQ (optional — graceful fallback if not installed)
# ---------------------------------------------------------------------------

try:
    from redis import Redis
    from rq import Queue as RQQueue
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False

_rq_queue:     Optional["RQQueue"]   = None
_thread_queue: Optional[ThreadQueue] = None
_mode: str = "uninitialized"


def _init_redis_queue() -> bool:
    """Try to connect to Redis and create the RQ queue. Returns True on success."""
    global _rq_queue, _mode
    try:
        conn = Redis(host="localhost", port=6379, socket_connect_timeout=2)
        conn.ping()  # raises if Redis is not running
        _rq_queue = RQQueue("monosplat", connection=conn)
        _mode = "redis"
        print("[queue] ✓  Redis connected — using RQ distributed queue")
        return True
    except Exception as e:
        print(f"[queue] ⚠  Redis not available ({e}) — falling back to thread queue")
        return False


def _init_thread_queue() -> None:
    """Initialize the in-process thread queue fallback."""
    global _thread_queue, _mode
    _thread_queue = ThreadQueue()
    _mode = "thread"
    print("[queue] ✓  Thread queue initialized (single-process mode)")


def initialize(prefer_redis: bool = True) -> str:
    """
    Initialize the queue system.

    Tries Redis/RQ first; falls back to thread queue.
    Returns "redis" or "thread" to indicate which mode is active.
    """
    if prefer_redis and _REDIS_AVAILABLE:
        if _init_redis_queue():
            return "redis"
    _init_thread_queue()
    return "thread"


def get_mode() -> str:
    return _mode


def enqueue_pipeline(fn, *args, job_timeout: int = 7200, **kwargs) -> Optional[str]:
    """
    Enqueue a pipeline function call.

    Args:
        fn:          Callable to run (must be importable by the worker).
        *args:       Positional arguments for fn.
        job_timeout: Max seconds before RQ kills the job (default 2 hours).
        **kwargs:    Keyword arguments for fn.

    Returns:
        job.id (str) when using Redis, or None in thread mode.
    """
    if _mode == "redis" and _rq_queue is not None:
        job = _rq_queue.enqueue(fn, *args, job_timeout=job_timeout, **kwargs)
        print(f"[queue] Enqueued RQ job {job.id}")
        return job.id

    if _mode == "thread" and _thread_queue is not None:
        # Run immediately in a daemon thread
        t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        t.start()
        print("[queue] Dispatched thread worker")
        return None

    raise RuntimeError(f"[queue] Queue not initialized (mode={_mode}). Call initialize() first.")


def get_rq_queue() -> Optional["RQQueue"]:
    """Return the raw RQ Queue object (None in thread mode)."""
    return _rq_queue
