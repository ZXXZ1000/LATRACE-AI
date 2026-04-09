from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters.ops.light_asd_scoring import (
    light_asd_status,
)
from modules.media_graph_compiler.adapters.ops.speaker_embedding import (
    extract_audio_embedding,
    warmup_speaker_embedding_model,
)


def _make_test_wav(path: Path, *, duration_s: float = 1.0, sample_rate: int = 16000) -> None:
    frame_count = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = []
        for i in range(frame_count):
            sample = int(0.2 * 32767.0 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            frames.append(struct.pack("<h", sample))
        wav.writeframes(b"".join(frames))


def main() -> None:
    speaker_warmup = warmup_speaker_embedding_model()
    speaker_vector_dim = 0
    with tempfile.TemporaryDirectory(prefix="mgc_bootstrap_") as tmp:
        wav_path = Path(tmp) / "warmup.wav"
        _make_test_wav(wav_path)
        speaker_vector_dim = len(extract_audio_embedding(str(wav_path)))

    report = {
        "speaker_embedding": {
            **speaker_warmup,
            "test_vector_dim": speaker_vector_dim,
        },
        "light_asd": light_asd_status(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
