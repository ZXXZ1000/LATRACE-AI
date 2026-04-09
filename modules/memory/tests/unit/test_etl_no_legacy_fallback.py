from __future__ import annotations

"""Ensure ETL no longer falls back to legacy mapping when VideoGraphMapper is unavailable.

We simulate the absence of VideoGraphMapper and expect run() to raise RuntimeError.
"""

import os
import pickle
from typing import Any, Dict

import pytest

import modules.memory.etl.pkl_to_db as etl


def _write_dummy_vg(path: str) -> None:
    # minimal VG dict composed of builtins so pickle works without custom classes
    vg: Dict[str, Any] = {
        "nodes": {
            1: {"type": "episodic", "metadata": {"timestamp": 1.0, "clip_id": 1}, "embeddings": None},
            2: {"type": "episodic", "metadata": {"timestamp": 2.0, "clip_id": 1}, "embeddings": None},
        },
        "edges": {},
    }
    with open(path, "wb") as f:
        pickle.dump(vg, f)


def test_etl_raises_when_mapper_absent(tmp_path: Any, monkeypatch: Any) -> None:
    # Arrange: write dummy vg pkl and simulate no mapper
    p = os.path.join(tmp_path, "vg.pkl")
    _write_dummy_vg(p)
    monkeypatch.setattr(etl, "VideoGraphMapper", None)

    # Act + Assert
    with pytest.raises(RuntimeError):
        etl.run(p, dry_run=True)
