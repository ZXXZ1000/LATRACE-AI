from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.application.window_scheduler import WindowScheduler
from modules.media_graph_compiler.types import WindowingPolicy


def test_window_scheduler_keeps_short_video_defaults() -> None:
    scheduler = WindowScheduler()
    policy = WindowingPolicy()

    settings = scheduler.resolve_settings(
        modality="video",
        clip_start_s=0.0,
        clip_end_s=12.0,
        policy=policy,
    )
    windows = scheduler.build_windows(
        modality="video",
        clip_start_s=0.0,
        clip_end_s=12.0,
        policy=policy,
        resolved_settings=settings,
    )

    assert settings["adaptive"] is False
    assert settings["window_size_seconds"] == 8.0
    assert settings["overlap_seconds"] == 2.0
    assert settings["estimated_window_count"] == 2
    assert len(windows) == 2


def test_window_scheduler_adapts_long_video_window_count() -> None:
    scheduler = WindowScheduler()
    policy = WindowingPolicy()

    settings = scheduler.resolve_settings(
        modality="video",
        clip_start_s=0.0,
        clip_end_s=302.0,
        policy=policy,
    )
    windows = scheduler.build_windows(
        modality="video",
        clip_start_s=0.0,
        clip_end_s=302.0,
        policy=policy,
        resolved_settings=settings,
    )

    assert settings["adaptive"] is True
    assert float(settings["window_size_seconds"]) > 8.0
    assert float(settings["overlap_seconds"]) <= 1.0
    assert int(settings["estimated_window_count"]) <= 32
    assert len(windows) == int(settings["estimated_window_count"])
