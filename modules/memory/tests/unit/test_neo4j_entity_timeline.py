from __future__ import annotations

import asyncio
from typing import List

from modules.memory.infra.neo4j_store import Neo4jStore


class _FakeSession:
    def __init__(self, results: List[list[dict]]) -> None:
        self._results = results

    def run(self, *args, **kwargs):
        if not self._results:
            return []
        return self._results.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, results: List[list[dict]]) -> None:
        self._results = results

    def session(self, database: str | None = None):
        return _FakeSession(self._results)


def test_query_entity_timeline_merges_evidence_and_utterance():
    store = Neo4jStore(settings={})
    store._driver = _FakeDriver(
        [
            [
                {
                    "segment_id": "seg-1",
                    "source_id": "demo.mp4",
                    "t_media_start": 1.0,
                    "t_media_end": 2.0,
                    "evidence_id": "ev-1",
                    "evidence_subtype": "face",
                    "confidence": 0.9,
                    "text": None,
                    "offset_in_segment": 0.1,
                    "utterance_id": None,
                }
            ],
            [
                {
                    "segment_id": "seg-1",
                    "source_id": "demo.mp4",
                    "t_media_start": 1.5,
                    "t_media_end": 2.1,
                    "utterance_id": "utt-1",
                    "raw_text": "hello",
                    "speaker_track_id": "spk-1",
                    "asr_model_version": "v1",
                    "lang": "en",
                }
            ],
        ]
    )

    res = asyncio.run(
        store.query_entity_timeline(
            tenant_id="t1",
            entity_id="person_1",
            limit=10,
        )
    )

    assert len(res) == 2
    assert res[0]["evidence_id"] == "ev-1"
    assert res[0]["kind"] == "evidence"
    assert res[1]["utterance_id"] == "utt-1"
    assert res[1]["kind"] == "utterance"
