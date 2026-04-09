from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters import (
    AudioFromVideoAdapter,
    MediaNormalizer,
    SemanticRuntime,
    VisualTrackRuntime,
)


class _Completed:
    def __init__(self, *, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Predictor:
    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "sess_1"}
        return {"frame_index": request.get("frame_index", 0), "outputs": {"out_obj_ids": [1]}}

    def handle_stream_request(self, request):
        yield {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": [1],
                "out_boxes_xywh": [[0, 0, 10, 10]],
                "out_binary_masks": ["mask_0"],
            },
        }
        yield {
            "frame_index": 1,
            "outputs": {
                "out_obj_ids": [1],
                "out_boxes_xywh": [[1, 1, 10, 10]],
                "out_binary_masks": ["mask_1"],
            },
        }


class _Processor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    def __call__(self, **kwargs):
        return kwargs


def _process_media(_messages, use_audio_in_video):
    return ["audio"], ["image"], ["video"]


def test_media_normalizer_returns_same_path_for_non_webm(tmp_path: Path) -> None:
    src = tmp_path / "demo.mp4"
    src.write_bytes(b"x")
    normalizer = MediaNormalizer()
    assert normalizer.ensure_video_path(src) == str(src)


def test_media_normalizer_runs_ffprobe(monkeypatch) -> None:
    normalizer = MediaNormalizer()

    def fake_run(cmd, check, capture_output, text):
        return _Completed(
            stdout='{"format":{"duration":"12.5"},"streams":[{"codec_type":"video","r_frame_rate":"25/1","width":1280,"height":720},{"codec_type":"audio"}]}'
        )

    monkeypatch.setattr("subprocess.run", fake_run)
    probe = normalizer.probe_media("/tmp/demo.mp4")
    assert probe["duration_seconds"] == 12.5
    assert probe["frame_rate"] == 25.0
    assert probe["has_audio"] is True


def test_audio_from_video_adapter_returns_output_path_on_success(monkeypatch, tmp_path: Path) -> None:
    adapter = AudioFromVideoAdapter()

    def fake_run(cmd, stdout, stderr):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"wav")
        return _Completed(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    out = adapter.extract_wav("/tmp/demo.mp4", output_dir=tmp_path)
    assert out is not None
    assert Path(out).exists()


def test_visual_track_runtime_converts_stream_outputs_to_stage_payload() -> None:
    runtime = VisualTrackRuntime(_Predictor())
    payload = runtime.run_text_prompt_tracking(
        resource_path="/tmp/demo.mp4",
        track_id="face_1",
        frame_timestamps_s=[0.0, 0.5],
    )
    assert payload["visual_tracks"][0]["track_id"] == "face_1"
    assert payload["visual_tracks"][0]["t_end_s"] == 0.5
    assert len(payload["evidence"]) == 2
    assert payload["evidence"][0]["metadata"]["track_id"] == "face_1"


def test_visual_track_runtime_supports_box_prompt_tracking() -> None:
    runtime = VisualTrackRuntime(_Predictor())
    payload = runtime.run_box_prompt_tracking(
        resource_path="/tmp/demo.mp4",
        boxes_xywh=[[10, 20, 30, 40]],
        prompt_frame_index=1,
        track_id="face_2",
        frame_timestamps_s=[0.0, 0.5],
        include_masks=True,
    )
    assert payload["visual_tracks"][0]["track_id"] == "face_2"
    assert payload["visual_tracks"][0]["frame_start"] == 0
    assert payload["evidence"][0]["metadata"]["runtime"] == "visual_tracking_box_prompt"
    assert payload["evidence"][0]["metadata"]["out_binary_masks"] == ["mask_0"]


def test_semantic_runtime_builds_vllm_inputs() -> None:
    runtime = SemanticRuntime(processor=_Processor(), process_media_fn=_process_media)
    payload = runtime.build_vllm_inputs(messages=[{"role": "user", "content": []}], use_audio_in_video=True)
    assert payload["prompt"] == "PROMPT"
    assert payload["multi_modal_data"]["audio"] == ["audio"]
