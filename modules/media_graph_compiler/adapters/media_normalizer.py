from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class MediaNormalizer:
    """Normalize local media files into a stable form for downstream stages.

    Adapted from the reference demo's `to_mp4(...)` helper and the older local
    FFmpeg extraction flow. The goal here is not to be clever. It is to make
    sure downstream stages always receive a predictable local path and
    best-effort probe metadata.
    """

    def ensure_video_path(self, path: str | Path) -> str:
        src = Path(path)
        if src.suffix.lower() != ".webm":
            return str(src)
        dst = src.with_suffix(".mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "fastdecode",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-threads",
            "0",
            "-f",
            "mp4",
            str(dst),
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return str(dst)

    def probe_media(self, path: str | Path) -> Dict[str, Any]:
        src = Path(path)
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,r_frame_rate,width,height",
            "-of",
            "json",
            str(src),
        ]
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams") or []
        duration = self._coerce_float((payload.get("format") or {}).get("duration"))
        frame_rate = 0.0
        width = None
        height = None
        has_audio = False
        for stream in streams:
            codec_type = stream.get("codec_type")
            if codec_type == "video":
                frame_rate = self._parse_frame_rate(stream.get("r_frame_rate"))
                width = stream.get("width")
                height = stream.get("height")
            if codec_type == "audio":
                has_audio = True
        return {
            "path": str(src),
            "duration_seconds": duration,
            "frame_rate": frame_rate,
            "width": width,
            "height": height,
            "has_audio": has_audio,
        }

    @staticmethod
    def _parse_frame_rate(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        raw = str(value)
        if "/" in raw:
            left, right = raw.split("/", 1)
            try:
                num = float(left)
                den = float(right)
                return num / den if den else 0.0
            except Exception:
                return 0.0
        try:
            return float(raw)
        except Exception:
            return 0.0

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None


__all__ = ["MediaNormalizer"]
