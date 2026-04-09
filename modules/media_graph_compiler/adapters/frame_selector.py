from __future__ import annotations

import hashlib
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class FrameSelectionResult:
    kept_indices: List[int]
    dropped_indices: List[int]
    hashes: Dict[int, str]


class FrameSelector:
    """Copy-first frame filtering utility borrowed from the legacy slice path.

    Why this exists:
    - `memorization_agent.step_slice` already proved that perceptual hash based
      dedup + uniform cap materially reduces wasted downstream compute.
    - We want the same policy to be available in the new media-first pipeline
      without dragging the old DAG back in.
    """

    def select_indices(
        self,
        frames: Sequence[str],
        *,
        enable_dedup: bool = True,
        similarity_threshold: int = 5,
        max_frames: int | None = None,
    ) -> FrameSelectionResult:
        kept_indices = list(range(len(frames)))
        hashes: Dict[int, str] = {}

        if enable_dedup and len(frames) > 2:
            kept_indices = []
            last_hash: Optional[str] = None
            for index, frame in enumerate(frames):
                current_hash = self._safe_hash(frame)
                if current_hash is not None:
                    hashes[index] = current_hash
                if (
                    last_hash is None
                    or current_hash is None
                    or not self._is_similar(last_hash, current_hash, threshold=similarity_threshold)
                ):
                    kept_indices.append(index)
                    last_hash = current_hash

        if max_frames is not None and max_frames > 0 and len(kept_indices) > max_frames:
            stride = max(1, int(len(kept_indices) / max_frames))
            kept_indices = kept_indices[::stride][:max_frames]

        kept_set = set(kept_indices)
        dropped_indices = [index for index in range(len(frames)) if index not in kept_set]
        return FrameSelectionResult(
            kept_indices=kept_indices,
            dropped_indices=dropped_indices,
            hashes=hashes,
        )

    @staticmethod
    def sample_indices(count: int, cap: int) -> List[int]:
        if cap <= 0 or count <= cap:
            return list(range(count))
        step = count / float(cap)
        return [int(i * step) for i in range(cap)]

    def _safe_hash(self, frame: str) -> Optional[str]:
        try:
            from modules.shared.lib.image_hash import average_hash, average_hash_b64
            from PIL import Image

            if isinstance(frame, str) and os.path.exists(frame):
                with Image.open(frame) as image:
                    return average_hash(image)
            return average_hash_b64(frame)
        except Exception:
            return self._fallback_hash(frame)

    @staticmethod
    def _is_similar(hash1: str, hash2: str, *, threshold: int) -> bool:
        try:
            from modules.shared.lib.image_hash import is_similar

            return is_similar(hash1, hash2, threshold=threshold)
        except Exception:
            return hash1 == hash2

    @staticmethod
    def _fallback_hash(frame: str) -> str:
        try:
            if isinstance(frame, str) and os.path.exists(frame):
                return hashlib.sha256(Path(frame).read_bytes()).hexdigest()[:16]
            return hashlib.sha256(str(frame).encode("utf-8")).hexdigest()[:16]
        except Exception:
            return ""


__all__ = ["FrameSelectionResult", "FrameSelector"]
