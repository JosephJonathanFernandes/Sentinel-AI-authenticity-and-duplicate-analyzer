"""SQLAlchemy database session and engine management.

Uses a file-based SQLite database. `check_same_thread=False` allows the
FastAPI / Streamlit frontends to share the engine safely across threads
when sessions are created per request.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base
from ..config import DATABASE_PATH


def get_database_url(db_path: Path | str | None = None) -> str:
    path = Path(db_path) if db_path is not None else DATABASE_PATH
    return f"sqlite:///{path}"


def create_engine_for(path: Path | str | None = None):
    url = get_database_url(path)
    # check_same_thread=False is required for SQLite when accessed from
    # multiple threads (e.g. FastAPI + Streamlit).
    engine = create_engine(url, connect_args={"check_same_thread": False}, echo=False)
    return engine


def init_db(engine) -> None:
    """Create all tables. Safe to call repeatedly (idempotent)."""
    Base.metadata.create_all(engine)


def session_factory(engine) -> sessionmaker[Session]:
    # A new sessionmaker per engine avoids cross-test / cross-database pollution
    # from a module-level cache. Creating a sessionmaker is inexpensive.
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def get_session(engine) -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and close it afterwards.

    Usage:
        with get_session(engine) as session:
            ...
    """
    factory = session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
