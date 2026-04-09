"""Pytest fixtures for memory module tests.

Provides centralized path management to avoid hardcoded paths.
"""
from __future__ import annotations

from pathlib import Path
import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory.

    Computed from this file's location:
    tests/conftest.py -> tests -> memory -> modules -> project_root
    """
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session")
def memory_dir(project_root: Path) -> Path:
    """Return memory module directory."""
    return project_root / "modules" / "memory"


@pytest.fixture(scope="session")
def config_dir(memory_dir: Path) -> Path:
    """Return memory config directory."""
    return memory_dir / "config"


@pytest.fixture(scope="session")
def api_dir(memory_dir: Path) -> Path:
    """Return memory API directory."""
    return memory_dir / "api"


@pytest.fixture(scope="session")
def test_dir(memory_dir: Path) -> Path:
    """Return memory tests directory."""
    return memory_dir / "tests"
