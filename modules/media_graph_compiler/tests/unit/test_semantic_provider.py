from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler import (
    CompileVideoRequest,
    MediaRoutingContext,
    MediaSourceRef,
)
from modules.media_graph_compiler.application.semantic_provider import (
    RichBatchSemanticProvider,
)


class _DummyAdapter:
    def __init__(self, *, kind: str = "openrouter_http") -> None:
        self.kind = kind
        self.last_messages = None
        self.last_response_format = None

    def generate(self, messages, response_format=None) -> str:
        self.last_messages = messages
        self.last_response_format = response_format
        return json.dumps(
            {
                "semantic_timeline": [
                    {
                        "text": "face_1 在厨房里说 hello world。",
                        "actor_tag": "face_1",
                        "images": ["img1"],
                    }
                ],
                "semantic": ["厨房里有人说话"],
                "equivalence": [["voice_1", "character_A"]],
            }
        )


def _request(tmp_path: Path, *, prompt_profile: str = "strict_json") -> CompileVideoRequest:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    return CompileVideoRequest(
        routing=MediaRoutingContext(
            tenant_id="tenant_local",
            user_id=["u:test"],
            memory_domain="media",
        ),
        source=MediaSourceRef(source_id="video_1", file_path=str(video_path)),
        metadata={"prompt_profile": prompt_profile},
    )


def test_semantic_provider_converts_file_path_images_to_data_urls(tmp_path: Path) -> None:
    adapter = _DummyAdapter(kind="openrouter_http")
    provider = RichBatchSemanticProvider(adapter=adapter)
    request = _request(tmp_path)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fh:
        fh.write(b"not-a-real-jpeg")
        fh.flush()
        image_path = fh.name

    try:
        out = provider.generate_window_digests(
            {
                "request": request,
                "optimization_plan": {"visual": {"max_frames_per_window": 12}},
                "window_payloads": [
                    {
                        "window_id": "w1",
                        "modality": "video",
                        "t_start_s": 0.0,
                        "t_end_s": 2.0,
                        "clip_frames": [
                            {"file_path": image_path, "frame_index": 0, "t_media_s": 0.5}
                        ],
                        "face_frames": [],
                        "visual_tracks": [{"track_id": "face_1"}],
                        "speaker_tracks": [{"track_id": "voice_1"}],
                        "utterances": [{"text": "hello world"}],
                        "evidence": [{"evidence_id": "ev1"}],
                    }
                ],
            }
        )
    finally:
        Path(image_path).unlink(missing_ok=True)

    assert out["window_digests"][0]["summary"]
    assert adapter.last_messages is not None
    user_message = next(msg for msg in adapter.last_messages if msg.get("role") == "user")
    parts = user_message.get("content") or []
    image_parts = [
        item for item in parts if isinstance(item, dict) and item.get("type") == "image_url"
    ]
    assert image_parts
    url = image_parts[0]["image_url"]["url"]
    assert isinstance(url, str) and url.startswith("data:image/")


def test_semantic_provider_uses_copied_prompt_profiles(tmp_path: Path) -> None:
    adapter = _DummyAdapter()
    provider = RichBatchSemanticProvider(adapter=adapter)
    request = _request(tmp_path, prompt_profile="rich_context")

    provider.generate_window_digests(
        {
            "request": request,
            "optimization_plan": {"visual": {"max_frames_per_window": 12}},
            "window_payloads": [
                {
                    "window_id": "w1",
                    "modality": "video",
                    "t_start_s": 0.0,
                    "t_end_s": 2.0,
                    "clip_frames": [],
                    "face_frames": [],
                    "visual_tracks": [],
                    "speaker_tracks": [],
                    "utterances": [],
                    "evidence": [],
                }
            ],
        }
    )

    assert adapter.last_messages is not None
    system_message = next(msg for msg in adapter.last_messages if msg.get("role") == "system")
    assert "Profile: rich_context" in str(system_message.get("content") or "")


def test_semantic_provider_falls_back_when_adapter_is_unavailable(tmp_path: Path) -> None:
    provider = RichBatchSemanticProvider()
    request = _request(tmp_path)

    out = provider.generate_window_digests(
        {
            "request": request,
            "optimization_plan": {"visual": {"max_frames_per_window": 12}},
            "window_payloads": [
                {
                    "window_id": "w1",
                    "modality": "audio",
                    "t_start_s": 0.0,
                    "t_end_s": 2.0,
                    "speaker_tracks": [{"track_id": "voice_1"}],
                    "utterances": [{"text": "hello world"}],
                    "evidence": [{"evidence_id": "ev1"}],
                }
            ],
        }
    )

    digest = out["window_digests"][0]
    assert "semantic_provider_not_configured" in digest["warnings"]
    assert digest["summary"] == "hello world"
