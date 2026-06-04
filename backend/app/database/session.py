"""
backend/app/database/session.py
--------------------------------
SQLAlchemy engine + session factory for MonoSplat backend.
Uses SQLite by default (file-based, zero-config); swap DATABASE_URL env
var to postgresql+asyncpg://... for production.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ── Database URL ──────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    """Walk up from this file until we find the repo root (contains configs/)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "configs").exists() and (parent / "src").exists():
            return parent
    # fallback: 4 levels up
    return p.parents[4]

_REPO_ROOT  = _find_repo_root()
_DEFAULT_DB = _REPO_ROOT / "data" / "monosplat.db"
# Ensure the data directory exists so SQLite can open the file
_DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL: str = os.getenv("MONOSPLAT_DATABASE_URL", f"sqlite:///{_DEFAULT_DB}")

# SQLite needs check_same_thread=False for multi-threaded FastAPI workers.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=bool(os.getenv("MONOSPLAT_SQL_ECHO", "")),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base for all MonoSplat ORM models."""


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_db():
    """Yield a database session; always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
