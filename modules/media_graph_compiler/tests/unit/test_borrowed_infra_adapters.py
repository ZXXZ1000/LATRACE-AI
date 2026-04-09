from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters import (
    MultimodalInputBuilder,
    VisualTrackingSessionRequestBuilder,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls.append(
            ("template", messages, tokenize, add_generation_prompt)
        )
        return "PROMPT"

    def __call__(self, **kwargs):
        self.calls.append(("processor", kwargs))
        return kwargs


class _FakePredictor:
    def __init__(self) -> None:
        self.requests = []

    def handle_request(self, request):
        self.requests.append(("request", request))
        if request["type"] == "start_session":
            return {"session_id": "session_1"}
        return {"frame_index": request.get("frame_index", 0), "outputs": {"ok": True}}

    def handle_stream_request(self, request):
        self.requests.append(("stream", request))
        yield {"frame_index": 0, "outputs": {"out_binary_masks": ["mask_1"]}}
        yield {"frame_index": 1, "outputs": {"out_binary_masks": ["mask_2"]}}


def _fake_process_mm_info(messages, use_audio_in_video):
    return ["audio_1"], ["image_1"], ["video_1"]


def test_qwen_builder_keeps_processor_and_media_split() -> None:
    builder = MultimodalInputBuilder()
    processor = _FakeProcessor()
    messages = [{"role": "user", "content": [{"type": "video", "video": "/tmp/demo.mp4"}]}]

    inputs = builder.build_transformers_inputs(
        processor=processor,
        process_mm_info=_fake_process_mm_info,
        messages=messages,
        use_audio_in_video=True,
    )

    assert inputs["text"] == "PROMPT"
    assert inputs["audio"] == ["audio_1"]
    assert inputs["images"] == ["image_1"]
    assert inputs["videos"] == ["video_1"]
    assert inputs["use_audio_in_video"] is True


def test_qwen_builder_produces_vllm_style_payload() -> None:
    builder = MultimodalInputBuilder()
    processor = _FakeProcessor()
    messages = [{"role": "user", "content": [{"type": "audio", "audio": "/tmp/demo.wav"}]}]

    inputs = builder.build_vllm_inputs(
        processor=processor,
        process_mm_info=_fake_process_mm_info,
        messages=messages,
        use_audio_in_video=False,
    )

    assert inputs["prompt"] == "PROMPT"
    assert inputs["multi_modal_data"]["audio"] == ["audio_1"]
    assert inputs["multi_modal_data"]["image"] == ["image_1"]
    assert inputs["multi_modal_data"]["video"] == ["video_1"]
    assert inputs["mm_processor_kwargs"]["use_audio_in_video"] is False


def test_visual_tracking_session_builder_matches_session_prompt_propagate_shape() -> None:
    builder = VisualTrackingSessionRequestBuilder()
    predictor = _FakePredictor()

    result = builder.run_text_tracking(
        predictor=predictor,
        resource_path="/tmp/frames_or_video.mp4",
        prompt_text="person",
        prompt_frame_index=3,
        propagation_direction="forward",
    )

    assert result["session_id"] == "session_1"
    assert result["prompt_response"]["frame_index"] == 3
    assert len(result["outputs"]) == 2
    assert predictor.requests[0][1]["type"] == "start_session"
    assert predictor.requests[1][1]["type"] == "add_prompt"
    assert predictor.requests[2][1]["type"] == "propagate_in_video"


def test_visual_tracking_session_builder_supports_box_prompt_tracking() -> None:
    builder = VisualTrackingSessionRequestBuilder()
    predictor = _FakePredictor()

    result = builder.run_box_tracking(
        predictor=predictor,
        resource_path="/tmp/frames_or_video.mp4",
        boxes_xywh=[[10, 20, 30, 40]],
        prompt_frame_index=2,
        labels=[1],
    )

    assert result["session_id"] == "session_1"
    assert len(result["outputs"]) == 2
    assert predictor.requests[0][1]["type"] == "start_session"
    assert predictor.requests[1][1]["bounding_boxes"] == [[10, 20, 30, 40]]
    assert predictor.requests[2][1]["type"] == "propagate_in_video"
