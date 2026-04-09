from __future__ import annotations

# 精简版音频分段与 ASR 提示（从旧算子直接迁移）
prompt_audio_segmentation = """You are given a video. Your task is to perform Automatic Speech Recognition (ASR) and audio diarization on the provided video. Extract all speech segments with accurate timestamps and segment them by speaker turns (i.e., different speakers should have separate segments), but without assigning speaker identifiers.

Return a JSON list where each entry represents a speech segment with the following fields:
  • start_time: Start timestamp in MM:SS format.
  • end_time: End timestamp in MM:SS format.
  • asr: The transcribed text for that segment.

Strict Requirements:
  • Ensure precise speech segmentation with accurate timestamps.
  • Segment based on speaker turns (i.e., different speakers' utterances should be separated).
  • Preserve punctuation and capitalization in the ASR output.
  • Skip the speeches that can hardly be clearly recognized.
  • Return only the valid JSON list (which starts with "[" and ends with "]") without additional explanations.
  • If the video contains no speech, return an empty list ("[]").
"""


__all__ = ["prompt_audio_segmentation"]
