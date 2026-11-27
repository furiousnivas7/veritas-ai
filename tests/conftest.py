from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.config import Settings
from veritas.core.db import make_engine, make_session_factory
from veritas.ingestion.service import IngestionService

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import make_sample_documents as samples  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", env="test", max_upload_mb=1)


@pytest.fixture
def sessions(settings: Settings) -> sessionmaker[Session]:
    return make_session_factory(make_engine(settings.resolved_database_url))


@pytest.fixture
def ingestion(settings: Settings, sessions: sessionmaker[Session]) -> IngestionService:
    return IngestionService(sessions, settings.documents_dir, settings.max_upload_mb * 1024 * 1024)


@pytest.fixture(scope="session")
def sample_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The three generated sample documents, keyed by format."""
    paths = samples.main(tmp_path_factory.mktemp("samples"))
    return {p.suffix.lstrip("."): p for p in paths}
