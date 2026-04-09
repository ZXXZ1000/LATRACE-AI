"""Migrated local media operators for media_graph_compiler.

These files are copied from `memorization_agent/ops` first, then adapted
minimally inside this module. Wrappers in `adapters/` should delegate here
instead of re-implementing core media logic.
"""

from modules.media_graph_compiler.adapters.ops.asr_local import (
    get_last_asr_status,
    transcribe_audio_b64,
)
from modules.media_graph_compiler.adapters.ops.audio_segments import (
    audio_duration_seconds,
    mean_embedding,
    slice_audio_b64_segments,
)
from modules.media_graph_compiler.adapters.ops.light_asd_scoring import (
    light_asd_status,
    score_light_asd,
)
from modules.media_graph_compiler.adapters.ops.speaker_diarization import (
    diarize_audio_b64,
    get_last_diarization_status,
)
from modules.media_graph_compiler.adapters.ops.speaker_embedding import (
    build_track_embedding,
    extract_audio_embedding,
)
from modules.media_graph_compiler.adapters.ops.voice_processing import (
    asr_transcribe_with_adapt,
    get_audio_embeddings,
    normalize_embedding,
    process_voices,
)
from modules.media_graph_compiler.adapters.ops.video_processing import (
    extract_frames,
    get_video_info,
    process_video_clip,
    process_video_to_fs,
)

__all__ = [
    "extract_frames",
    "get_video_info",
    "process_video_clip",
    "process_video_to_fs",
    "audio_duration_seconds",
    "asr_transcribe_with_adapt",
    "get_last_asr_status",
    "build_track_embedding",
    "diarize_audio_b64",
    "get_last_diarization_status",
    "extract_audio_embedding",
    "get_audio_embeddings",
    "light_asd_status",
    "mean_embedding",
    "normalize_embedding",
    "process_voices",
    "score_light_asd",
    "slice_audio_b64_segments",
    "transcribe_audio_b64",
]
