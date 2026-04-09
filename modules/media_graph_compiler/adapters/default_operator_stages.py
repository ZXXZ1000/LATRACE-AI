from __future__ import annotations

from typing import Any, Dict, Mapping

from modules.media_graph_compiler.application.stage_names import (
    FACE_VOICE_ASSOCIATION_STAGE,
    LEGACY_ASSOCIATION_STAGE,
    LEGACY_SEMANTIC_STAGE,
    LEGACY_SPEAKER_STAGE,
    LEGACY_VISUAL_STAGE,
    SEMANTIC_COMPILE_STAGE,
    SPEAKER_TRACK_STAGE,
    VISUAL_TRACK_STAGE,
)
from modules.media_graph_compiler.adapters.local_face_voice_association_stage import (
    LocalFaceVoiceAssociationStage,
)
from modules.media_graph_compiler.adapters.local_semantic_stage import (
    LocalSemanticStage,
)
from modules.media_graph_compiler.adapters.local_speaker_stage import LocalSpeakerStage
from modules.media_graph_compiler.adapters.local_visual_stage import LocalVisualStage

_LOCAL_VISUAL_STAGE: LocalVisualStage | None = None
_LOCAL_SPEAKER_STAGE: LocalSpeakerStage | None = None
_LOCAL_FACE_VOICE_ASSOCIATION_STAGE: LocalFaceVoiceAssociationStage | None = None
_LOCAL_SEMANTIC_STAGE: LocalSemanticStage | None = None


def _get_visual_stage() -> LocalVisualStage:
    global _LOCAL_VISUAL_STAGE
    if _LOCAL_VISUAL_STAGE is None:
        _LOCAL_VISUAL_STAGE = LocalVisualStage()
    return _LOCAL_VISUAL_STAGE


def _get_speaker_stage() -> LocalSpeakerStage:
    global _LOCAL_SPEAKER_STAGE
    if _LOCAL_SPEAKER_STAGE is None:
        _LOCAL_SPEAKER_STAGE = LocalSpeakerStage()
    return _LOCAL_SPEAKER_STAGE


def _get_face_voice_association_stage() -> LocalFaceVoiceAssociationStage:
    global _LOCAL_FACE_VOICE_ASSOCIATION_STAGE
    if _LOCAL_FACE_VOICE_ASSOCIATION_STAGE is None:
        _LOCAL_FACE_VOICE_ASSOCIATION_STAGE = LocalFaceVoiceAssociationStage()
    return _LOCAL_FACE_VOICE_ASSOCIATION_STAGE


def _get_semantic_stage() -> LocalSemanticStage:
    global _LOCAL_SEMANTIC_STAGE
    if _LOCAL_SEMANTIC_STAGE is None:
        _LOCAL_SEMANTIC_STAGE = LocalSemanticStage()
    return _LOCAL_SEMANTIC_STAGE


def _visual_stage(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    return _get_visual_stage().run(ctx)


def _speaker_stage(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    return _get_speaker_stage().run(ctx)


def _face_voice_association_stage(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    return _get_face_voice_association_stage().run(ctx)


def _semantic_stage(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    return _get_semantic_stage().run(ctx)


def ensure_default_operator_stages(operator_bus) -> None:
    if operator_bus.get(VISUAL_TRACK_STAGE) is None and operator_bus.get(LEGACY_VISUAL_STAGE) is None:
        operator_bus.register(VISUAL_TRACK_STAGE, _visual_stage)
    if operator_bus.get(SPEAKER_TRACK_STAGE) is None and operator_bus.get(LEGACY_SPEAKER_STAGE) is None:
        operator_bus.register(SPEAKER_TRACK_STAGE, _speaker_stage)
    if operator_bus.get(FACE_VOICE_ASSOCIATION_STAGE) is None and operator_bus.get(LEGACY_ASSOCIATION_STAGE) is None:
        operator_bus.register(FACE_VOICE_ASSOCIATION_STAGE, _face_voice_association_stage)
    if operator_bus.get(SEMANTIC_COMPILE_STAGE) is None and operator_bus.get(LEGACY_SEMANTIC_STAGE) is None:
        operator_bus.register(SEMANTIC_COMPILE_STAGE, _semantic_stage)


__all__ = ["ensure_default_operator_stages"]
