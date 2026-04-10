from __future__ import annotations

import json
import threading
import tempfile
import time
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


class _RetryAdapter:
    def __init__(self) -> None:
        self.kind = "openrouter_http"
        self.response_formats = []
        self.calls = 0

    def generate(self, messages, response_format=None) -> str:
        self.calls += 1
        self.response_formats.append(response_format)
        if self.calls == 1:
            return "not-json-at-all"
        return json.dumps(
            {
                "summary": "retry succeeded",
                "semantic_timeline": [{"text": "retry succeeded"}],
            }
        )


class _DelayedAdapter:
    def __init__(self, delays: dict[str, float]) -> None:
        self.kind = "openrouter_http"
        self._delays = dict(delays)
        self._lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def generate(self, messages, response_format=None) -> str:
        window_id = "unknown"
        for message in messages:
            if message.get("role") != "user":
                continue
            for item in message.get("content") or []:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = str(item.get("text") or "")
                if not text.startswith("{"):
                    continue
                parsed = json.loads(text)
                window_id = str(parsed.get("window_id") or "unknown")
                break
            if window_id != "unknown":
                break
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(float(self._delays.get(window_id, 0.0)))
            return json.dumps(
                {
                    "summary": f"{window_id} summary",
                    "semantic_timeline": [{"text": f"{window_id} summary"}],
                }
            )
        finally:
            with self._lock:
                self.active_calls -= 1


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


def test_semantic_provider_shrinks_context_and_caps_openrouter_images(tmp_path: Path) -> None:
    adapter = _DummyAdapter(kind="openrouter_http")
    provider = RichBatchSemanticProvider(adapter=adapter)
    request = _request(tmp_path)

    frame_paths = []
    for index in range(4):
        image_path = tmp_path / f"frame_{index}.jpg"
        image_path.write_bytes(f"fake-image-{index}".encode("utf-8"))
        frame_paths.append(str(image_path))

    provider.generate_window_digests(
        {
            "request": request,
            "optimization_plan": {"visual": {"max_frames_per_window": 12}},
            "window_payloads": [
                {
                    "window_id": "w1",
                    "modality": "video",
                    "t_start_s": 0.0,
                    "t_end_s": 8.0,
                    "clip_frames": [
                        {"file_path": path, "frame_index": index, "t_media_s": float(index)}
                        for index, path in enumerate(frame_paths)
                    ],
                    "representative_frames": [
                        {"file_path": path, "frame_index": index, "t_media_s": float(index)}
                        for index, path in enumerate(frame_paths)
                    ],
                    "segment_visual_profile": {
                        "representative_thumbnails": [
                            {"frame_index": index, "t_media_s": float(index)}
                            for index in range(4)
                        ],
                        "thumbnail_evidence_refs": ["ev1", "ev2", "ev3", "ev4"],
                        "vector_summary": {
                            "provider": "hash",
                            "strategy": "mean_pool",
                            "sample_count": 4,
                            "dim": 512,
                            "preview": list(range(16)),
                        },
                    },
                    "window_stats": {
                        "clip_frames_total": 4,
                        "clip_frames_selected": 4,
                        "representative_frames": 4,
                        "visual_tracks": 6,
                        "speaker_tracks": 5,
                        "face_voice_links": 5,
                        "utterances": 12,
                        "evidence": 12,
                    },
                    "visual_tracks": [{"track_id": f"face_{i}", "t_start_s": 0.0, "t_end_s": 4.0} for i in range(6)],
                    "speaker_tracks": [{"track_id": f"voice_{i}", "t_start_s": 0.0, "t_end_s": 4.0} for i in range(5)],
                    "face_voice_links": [
                        {
                            "link_id": f"link_{i}",
                            "speaker_track_id": f"voice_{i}",
                            "visual_track_id": f"face_{i}",
                            "confidence": 0.9 - (i * 0.1),
                        }
                        for i in range(5)
                    ],
                    "utterances": [
                        {
                            "utterance_id": f"utt_{i}",
                            "speaker_track_id": "voice_1",
                            "t_start_s": float(i),
                            "t_end_s": float(i) + 0.5,
                            "text": f"utterance {i}",
                        }
                        for i in range(12)
                    ],
                    "evidence": [
                        {
                            "evidence_id": f"ev_{i}",
                            "kind": "frame_crop" if i < 4 else "transcript",
                            "t_start_s": float(i),
                            "t_end_s": float(i) + 0.5,
                            "metadata": {"transcript": f"line {i}", "frame_index": i},
                        }
                        for i in range(12)
                    ],
                }
            ],
        }
    )

    assert adapter.last_messages is not None
    user_message = next(msg for msg in adapter.last_messages if msg.get("role") == "user")
    parts = user_message.get("content") or []
    image_parts = [item for item in parts if isinstance(item, dict) and item.get("type") == "image_url"]
    assert len(image_parts) == 1
    structured = json.loads(
        next(
            item["text"]
            for item in parts
            if isinstance(item, dict)
            and item.get("type") == "text"
            and str(item.get("text") or "").startswith("{")
        )
    )
    assert structured["window_stats"]["representative_frames"] == 4
    assert len(structured["visual_tracks"]) == 4
    assert len(structured["speaker_tracks"]) == 4
    assert len(structured["face_voice_links"]) == 4
    assert len(structured["utterances"]) == 6
    assert len(structured["evidence"]) == 6
    assert len(structured["segment_visual_profile"]["vector_summary"]["preview"]) == 8


def test_semantic_provider_honors_explicit_multi_image_override(tmp_path: Path) -> None:
    adapter = _DummyAdapter(kind="openrouter_http")
    provider = RichBatchSemanticProvider(adapter=adapter)
    request = _request(tmp_path)
    request.metadata["attach_frames"] = 3

    frame_paths = []
    for index in range(3):
        image_path = tmp_path / f"explicit_{index}.jpg"
        image_path.write_bytes(f"explicit-{index}".encode("utf-8"))
        frame_paths.append(str(image_path))

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
                    "clip_frames": [
                        {"file_path": path, "frame_index": index, "t_media_s": float(index)}
                        for index, path in enumerate(frame_paths)
                    ],
                    "representative_frames": [
                        {"file_path": path, "frame_index": index, "t_media_s": float(index)}
                        for index, path in enumerate(frame_paths)
                    ],
                    "visual_tracks": [],
                    "speaker_tracks": [],
                    "utterances": [],
                    "evidence": [],
                }
            ],
        }
    )

    user_message = next(msg for msg in adapter.last_messages if msg.get("role") == "user")
    parts = user_message.get("content") or []
    image_parts = [item for item in parts if isinstance(item, dict) and item.get("type") == "image_url"]
    assert len(image_parts) == 3


def test_semantic_provider_emits_face_voice_binding_hint(tmp_path: Path) -> None:
    adapter = _DummyAdapter(kind="openrouter_http")
    provider = RichBatchSemanticProvider(adapter=adapter)
    request = _request(tmp_path)

    provider.generate_window_digests(
        {
            "request": request,
            "optimization_plan": {"visual": {"max_frames_per_window": 12}},
            "window_payloads": [
                {
                    "window_id": "w1",
                    "modality": "video",
                    "t_start_s": 0.0,
                    "t_end_s": 8.0,
                    "clip_frames": [],
                    "face_frames": [],
                    "visual_tracks": [{"track_id": "face_1"}],
                    "speaker_tracks": [{"track_id": "voice_1"}],
                    "face_voice_links": [
                        {
                            "link_id": "link_1",
                            "speaker_track_id": "voice_1",
                            "visual_track_id": "face_1",
                            "confidence": 0.54,
                            "metadata": {"method": "light_asd_temporal_fusion"},
                        }
                    ],
                    "utterances": [
                        {
                            "utterance_id": "utt_1",
                            "speaker_track_id": "voice_1",
                            "t_start_s": 0.0,
                            "t_end_s": 1.0,
                            "text": "hello world",
                        }
                    ],
                    "evidence": [],
                }
            ],
        }
    )

    user_message = next(msg for msg in adapter.last_messages if msg.get("role") == "user")
    parts = user_message.get("content") or []
    hint_texts = [
        str(item.get("text") or "")
        for item in parts
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    hint = next(text for text in hint_texts if "声脸绑定提示：" in text)
    assert "voice_1->face_1" in hint
    assert "actor_tag 也优先用 face_#" in hint


def test_semantic_provider_retries_unparseable_response_once(tmp_path: Path) -> None:
    adapter = _RetryAdapter()
    provider = RichBatchSemanticProvider(adapter=adapter)
    request = _request(tmp_path)
    request.metadata["semantic_parse_max_attempts"] = 2

    out = provider.generate_window_digests(
        {
            "request": request,
            "optimization_plan": {"visual": {"max_frames_per_window": 12}},
            "window_payloads": [
                {
                    "window_id": "w_retry",
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

    digest = out["window_digests"][0]
    assert digest["summary"] == "retry succeeded"
    assert digest["semantic_payload"]["request_attempts"] == 2
    assert adapter.response_formats == [{"type": "json_object"}, None]


def test_semantic_provider_preserves_window_order_under_concurrency(tmp_path: Path) -> None:
    adapter = _DelayedAdapter(
        {
            "w1": 0.06,
            "w2": 0.02,
            "w3": 0.0,
        }
    )
    provider = RichBatchSemanticProvider(adapter=adapter)
    request = _request(tmp_path)
    request.metadata["semantic_max_concurrent"] = 3

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
                    "clip_frames": [],
                    "face_frames": [],
                    "visual_tracks": [],
                    "speaker_tracks": [],
                    "utterances": [],
                    "evidence": [],
                },
                {
                    "window_id": "w2",
                    "modality": "video",
                    "t_start_s": 2.0,
                    "t_end_s": 4.0,
                    "clip_frames": [],
                    "face_frames": [],
                    "visual_tracks": [],
                    "speaker_tracks": [],
                    "utterances": [],
                    "evidence": [],
                },
                {
                    "window_id": "w3",
                    "modality": "video",
                    "t_start_s": 4.0,
                    "t_end_s": 6.0,
                    "clip_frames": [],
                    "face_frames": [],
                    "visual_tracks": [],
                    "speaker_tracks": [],
                    "utterances": [],
                    "evidence": [],
                },
            ],
        }
    )

    assert [item["window_id"] for item in out["window_digests"]] == ["w1", "w2", "w3"]
    assert [item["summary"] for item in out["window_digests"]] == [
        "w1 summary",
        "w2 summary",
        "w3 summary",
    ]
    assert adapter.max_active_calls >= 2
