from __future__ import annotations

import numpy as np

from modules.media_graph_compiler.adapters.ops import speaker_embedding


def test_extract_audio_embedding_flattens_2d_vectors(monkeypatch) -> None:
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.ops.speaker_embedding._ensure_runtime",
        lambda: ("wespeakerruntime", type("R", (), {"extract_embedding": lambda self, _: np.ones((1, 4), dtype=np.float32)})()),
    )
    out = speaker_embedding.extract_audio_embedding("/tmp/demo.wav")
    assert out == [1.0, 1.0, 1.0, 1.0]
