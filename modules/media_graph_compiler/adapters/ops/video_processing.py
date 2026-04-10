from __future__ import annotations

# Copyright (2025) Bytedance Ltd. and/or its affiliates

import base64
import logging
import os
import subprocess
import tempfile

logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("moviepy.video.io.VideoFileClip").setLevel(logging.ERROR)
logging.getLogger("moviepy.audio.io.AudioFileClip").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def get_video_info(file_path):
    """Copied from the legacy video operator with minimal adaptation."""
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:  # pragma: no cover
        from moviepy import VideoFileClip

    file_info = {}
    file_info["path"] = file_path
    file_info["name"] = file_path.split("/")[-1]
    file_info["format"] = os.path.splitext(file_path)[1][1:].lower()

    video = VideoFileClip(file_path)
    file_info["fps"] = video.fps
    file_info["frames"] = int(video.fps * video.duration)
    file_info["duration"] = video.duration
    file_info["width"] = video.size[0]
    file_info["height"] = video.size[1]
    video.close()
    return file_info


def extract_frames(video, start_time=None, interval=None, sample_fps=10):
    import cv2
    import numpy as np

    if start_time is None and interval is None:
        start_time = 0
        interval = video.duration

    try:
        sample_fps = float(sample_fps)
    except Exception:
        sample_fps = 10.0
    if sample_fps <= 0:
        sample_fps = max(video.fps or 1.0, 1.0)

    frames = []
    frame_interval = 1.0 / sample_fps
    for t in np.arange(
        start_time,
        min(start_time + interval, video.duration),
        frame_interval,
    ):
        frame = video.get_frame(t)
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    return frames


def process_video_clip(video_path, fps=5, audio_fps=16000):
    try:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:  # pragma: no cover
            from moviepy import VideoFileClip

        base64_data = {}
        video = VideoFileClip(video_path)
        logger.info(
            "process_video_clip: path=%s duration=%.3fs fps=%.3f audio=%s",
            video_path,
            video.duration or 0.0,
            video.fps or 0.0,
            "yes" if video.audio else "no",
        )
        base64_data["video"] = base64.b64encode(open(video_path, "rb").read())
        base64_data["frames"] = extract_frames(video, sample_fps=fps)

        if not base64_data["frames"]:
            logger.warning(
                "process_video_clip: no frames extracted via sample_fps=%s, falling back to sequential frames",
                fps,
            )
            fallback_frames = []
            max_frames = int((video.fps or 1.0) * video.duration) if video.duration else 0
            max_frames = max(max_frames, 30)
            for idx, frame in enumerate(video.iter_frames()):
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                fallback_frames.append(base64.b64encode(buffer).decode("utf-8"))
                if idx + 1 >= max_frames:
                    break
            base64_data["frames"] = fallback_frames
        if not base64_data["frames"]:
            raise RuntimeError("process_video_clip: failed to extract any frames from video after fallback")

        if video.audio is None:
            base64_data["audio"] = None
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as audio_tempfile:
                video.audio.write_audiofile(audio_tempfile.name, codec="pcm_s16le", fps=audio_fps)
                audio_tempfile.seek(0)
                base64_data["audio"] = base64.b64encode(audio_tempfile.read())

        video.close()
        return base64_data["video"], base64_data["frames"], base64_data["audio"]

    except Exception as exc:
        logger.error("Error processing video clip: %s", str(exc))
        raise


def process_video_to_fs(
    video_path: str,
    *,
    fps: float = 0.5,
    clip_px: int = 256,
    face_px: int = 640,
    out_base: str = ".artifacts/media_graph_compiler/frames",
    audio_fps: int = 16000,
    clip_start_s: float = 0.0,
    clip_end_s: float | None = None,
) -> dict:
    """Copied from the legacy FFmpeg dual-stream extractor."""
    os.makedirs(out_base, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    uniq_dir = os.path.join(out_base, f"{base_name}_{os.getpid()}")
    clip_dir = os.path.join(uniq_dir, "clip")
    face_dir = os.path.join(uniq_dir, "face")
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)

    filter_complex = (
        f"[0:v]fps={float(fps)},split=2[s1][s2];"
        f"[s1]scale={int(clip_px)}:-1:flags=bilinear[out1];"
        f"[s2]scale={int(face_px)}:-1:flags=bilinear[out2]"
    )
    clip_start_s = max(0.0, float(clip_start_s or 0.0))
    clip_duration_s: float | None = None
    if clip_end_s is not None:
        try:
            clip_end_value = float(clip_end_s)
            if clip_end_value > clip_start_s:
                clip_duration_s = clip_end_value - clip_start_s
        except Exception:
            clip_duration_s = None

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if clip_start_s > 0.0:
        cmd.extend(["-ss", f"{clip_start_s:.3f}"])
    cmd.extend(["-i", video_path])
    if clip_duration_s is not None and clip_duration_s > 0.0:
        cmd.extend(["-t", f"{clip_duration_s:.3f}"])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out1]",
            os.path.join(clip_dir, "%06d.jpg"),
            "-map",
            "[out2]",
            os.path.join(face_dir, "%06d.jpg"),
        ]
    )
    try:
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            raise RuntimeError(ret.stderr.decode("utf-8", errors="ignore"))
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found in PATH") from exc

    import glob

    frames_clip = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
    frames_face = sorted(glob.glob(os.path.join(face_dir, "*.jpg")))
    if not frames_clip or not frames_face:
        raise RuntimeError("no frames extracted via ffmpeg")

    try:
        info = get_video_info(video_path)
        duration = float(info.get("duration") or 0.0)
        if clip_duration_s is not None and clip_duration_s > 0.0:
            duration = min(duration, clip_duration_s) if duration > 0.0 else clip_duration_s
    except Exception:
        duration = clip_duration_s or 0.0

    audio_b64 = None
    audio_path = None
    try:
        audio_dir = os.path.join(uniq_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"{base_name}.wav")
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            acmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
            ]
            if clip_start_s > 0.0:
                acmd.extend(["-ss", f"{clip_start_s:.3f}"])
            acmd.extend(["-i", video_path])
            if clip_duration_s is not None and clip_duration_s > 0.0:
                acmd.extend(["-t", f"{clip_duration_s:.3f}"])
            acmd.extend(
                [
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(int(audio_fps)),
                    tmp.name,
                ]
            )
            ares = subprocess.run(acmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ares.returncode == 0:
                raw = open(tmp.name, "rb").read()
                audio_b64 = base64.b64encode(raw)
                with open(audio_path, "wb") as f:
                    f.write(raw)
            else:
                audio_path = None
    except Exception:
        audio_b64 = None
        audio_path = None

    return {
        "frames_clip": frames_clip,
        "frames_face": frames_face,
        "audio_b64": audio_b64,
        "audio_path": audio_path,
        "duration": duration,
    }


__all__ = [
    "extract_frames",
    "get_video_info",
    "process_video_clip",
    "process_video_to_fs",
]
