"""Transcript normalization shared by every post-processing agent.

This is lifted verbatim out of the old minutes.py so both the minutes agent
and the action-items agent normalize/render segments the same way instead of
each keeping their own copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class TranscriptError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    index: int
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None


def normalize_transcript(raw: Any) -> list[Segment]:
    """Accept the common audio-analysis shapes, a segment list, or plain text."""
    if isinstance(raw, dict):
        raw = next(
            (raw[key] for key in ("segments", "results", "data", "transcript") if key in raw),
            [],
        )
        if isinstance(raw, dict):
            return normalize_transcript(raw)
    if isinstance(raw, str):
        raw = [{"speaker": "SPEAKER_00", "text": raw}]
    if not isinstance(raw, (list, tuple)):
        raise TranscriptError("Transcript must be a segment list, transcript object, or string.")

    segments: list[Segment] = []
    for item in raw:
        if isinstance(item, Segment):
            text, speaker, start, end = item.text, item.speaker, item.start, item.end
        elif isinstance(item, str):
            text, speaker, start, end = item, "SPEAKER_00", None, None
        elif isinstance(item, dict):
            text = next(
                (item[key] for key in ("text", "transcript", "content") if item.get(key) is not None),
                "",
            )
            speaker = next(
                (item[key] for key in ("speaker_id", "speaker", "label") if item.get(key) is not None),
                "SPEAKER_UNK",
            )
            start = item.get("start", item.get("start_time"))
            end = item.get("end", item.get("end_time"))
        else:
            continue

        text = " ".join(str(text).split())
        if text:
            segments.append(
                Segment(
                    index=len(segments),
                    speaker=str(speaker),
                    text=text,
                    start=float(start) if start is not None else None,
                    end=float(end) if end is not None else None,
                )
            )
    if not segments:
        raise TranscriptError("The transcript contains no non-empty segments.")
    return segments


def load_transcript(path: str | Path) -> list[Segment]:
    return normalize_transcript(json.loads(Path(path).read_text(encoding="utf-8")))


# def render_segments(segments: Sequence[Segment]) -> str:
#     return "\n".join(f"[{item.index:05d}] {item.speaker}: {item.text}" for item in segments)
def render_segments(segments: Sequence[Segment]) -> str:
    lines = []

    for item in segments:
        if item.start is not None and item.end is not None:
            timestamp = f"[{item.start:.2f}s - {item.end:.2f}s]"
        elif item.start is not None:
            timestamp = f"[{item.start:.2f}s]"
        else:
            timestamp = ""

        lines.append(
            f"[{item.index:05d}] {timestamp} {item.speaker}: {item.text}"
        )

    return "\n".join(lines)

def speaker_ids(segments: Sequence[Segment]) -> list[str]:
    return sorted({segment.speaker for segment in segments})


def fallback_names(ids: Sequence[str]) -> dict[str, str]:
    return {speaker_id: f"Speaker {chr(65 + index)}" for index, speaker_id in enumerate(ids)}