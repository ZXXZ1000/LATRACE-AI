from __future__ import annotations

import subprocess
from pathlib import Path


class AudioFromVideoAdapter:
    """Extract a local WAV track from a local video path using FFmpeg.

    Adapted directly from the older video-processing helper. Keep it dumb and
    predictable.
    """

    def extract_wav(
        self,
        video_path: str | Path,
        *,
        output_dir: str | Path,
        sample_rate: int = 16000,
    ) -> str | None:
        src = Path(video_path)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        dst = out_root / f"{src.stem}.wav"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(int(sample_rate)),
            str(dst),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            return None
        return str(dst)


__all__ = ["AudioFromVideoAdapter"]
