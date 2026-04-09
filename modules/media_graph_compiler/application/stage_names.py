"""Canonical internal stage names for media graph compilation.

These names are intentionally product-neutral. Public APIs expose only
`compile_video` and `compile_audio`; internal stage names remain swappable.
"""

VISUAL_TRACK_STAGE = "visual_track_stage"
SPEAKER_TRACK_STAGE = "speaker_track_stage"
FACE_VOICE_ASSOCIATION_STAGE = "face_voice_association_stage"
SEMANTIC_COMPILE_STAGE = "semantic_compile_stage"

# Backward-compatible aliases used by early drafts/tests.
LEGACY_VISUAL_STAGE = "visual_stage"
LEGACY_SPEAKER_STAGE = "speaker_stage"
LEGACY_ASSOCIATION_STAGE = "association_stage"
LEGACY_SEMANTIC_STAGE = "semantic_stage"

__all__ = [
    "VISUAL_TRACK_STAGE",
    "SPEAKER_TRACK_STAGE",
    "FACE_VOICE_ASSOCIATION_STAGE",
    "SEMANTIC_COMPILE_STAGE",
    "LEGACY_VISUAL_STAGE",
    "LEGACY_SPEAKER_STAGE",
    "LEGACY_ASSOCIATION_STAGE",
    "LEGACY_SEMANTIC_STAGE",
]
