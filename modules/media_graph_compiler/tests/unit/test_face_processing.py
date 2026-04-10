from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters.ops import face_processing


def test_face_processing_limits_insightface_modules_by_default(monkeypatch) -> None:
    calls = {}
    fake_module = ModuleType("insightface.app")

    class _FakeFaceAnalysis:
        def __init__(self, name="buffalo_l", providers=None, allowed_modules=None, **kwargs):
            calls["name"] = name
            calls["providers"] = list(providers or [])
            calls["allowed_modules"] = list(allowed_modules or [])

        def prepare(self, ctx_id=-1, providers=None):
            calls["ctx_id"] = ctx_id
            calls["prepare_providers"] = list(providers or [])

    fake_module.FaceAnalysis = _FakeFaceAnalysis
    monkeypatch.setitem(sys.modules, "insightface.app", fake_module)
    monkeypatch.delenv("MGC_FACE_ALLOWED_MODULES", raising=False)
    monkeypatch.setattr(face_processing, "_FACE_APP", None)
    monkeypatch.setattr(face_processing, "_FACE_APP_CONFIG_KEY", None)

    face_processing._ensure_face_app()

    assert calls["name"] == "buffalo_l"
    assert calls["allowed_modules"] == ["detection", "recognition"]
    assert calls["providers"]


def test_face_processing_rebuilds_runtime_when_allowed_modules_change(monkeypatch) -> None:
    init_calls = []
    fake_module = ModuleType("insightface.app")

    class _FakeFaceAnalysis:
        def __init__(self, name="buffalo_l", providers=None, allowed_modules=None, **kwargs):
            init_calls.append(list(allowed_modules or []))

        def prepare(self, ctx_id=-1, providers=None):
            return None

    fake_module.FaceAnalysis = _FakeFaceAnalysis
    monkeypatch.setitem(sys.modules, "insightface.app", fake_module)
    monkeypatch.setattr(face_processing, "_FACE_APP", None)
    monkeypatch.setattr(face_processing, "_FACE_APP_CONFIG_KEY", None)

    monkeypatch.setenv("MGC_FACE_ALLOWED_MODULES", "detection,recognition")
    face_processing._ensure_face_app()

    monkeypatch.setenv("MGC_FACE_ALLOWED_MODULES", "detection")
    face_processing._ensure_face_app()

    assert init_calls == [["detection", "recognition"], ["detection"]]
