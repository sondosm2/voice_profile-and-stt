"""Minutes extraction for upload, finished recording, and live-stream inputs.

The public surface is intentionally small: construct ``MinutesAgent`` with any
``StructuredLLM``, call ``run`` for a complete transcript, or use ``stream`` and feed
finalized windows. All chunking/map/reduce mechanics live in ``mapreduce.MapReduceJob``;
this file only holds what's actually meeting-specific: the prompts, speaker-name
resolution, and attaching the resolved roster to the final minutes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ..llms import CallMetric, StructuredLLM, TokenBudgetError
from ..map_reduce import MapReduceJob
from ..transcript import (
    Segment,
    fallback_names,
    load_transcript,
    normalize_transcript,
    render_segments,
    speaker_ids,
)
from .models import (
    MinutesOfMeeting,
    Participant,
    PartialMinutes,
    SpeakerIdentity,
    SpeakerRoster,
)

if TYPE_CHECKING:
    from .rendering import MinutesArtifacts


MOM_PROMPT = """You extract Minutes of Meeting for Idraak from a diarized transcript.
Use only supported facts and write free text in the meeting's dominant language. Preserve names.
Decisions are settled choices, not proposals or tasks. Open questions include key explicit questions;
use empty answer and answered_by values when unanswered. Topics are distinct main subjects. Issues are
explicit problems, risks, blockers, complaints, or constraints. Important notes are concrete facts,
figures, dates, scope statements, or announcements for the administrator. Do not create action-plan
tasks because a separate agent will do that. Prefer short, non-duplicated items and valid empty lists."""

SPEAKER_PROMPT = """Resolve diarization labels to people using only self-introductions, direct address,
handovers, and explicit role clues. Return one entry per supplied speaker_id. Use an empty name or role
when unsupported. Preserve the original script and never assign one name to multiple labels."""

PARTIAL_REDUCE_PROMPT = """Merge chronological partial meeting extractions. Preserve every distinct
grounded fact and the dominant language, merge duplicates, keep only settled decisions, and invent
nothing. Return another partial extraction."""

FINAL_REDUCE_PROMPT = """Merge chronological partial Minutes of Meeting into one final document.
Preserve every distinct grounded fact, merge duplicates, keep only settled decisions, and invent
nothing. Use the supplied resolved roster for exactly one participant per diarization label."""


class TranscriptError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionPlan:
    segments: int
    estimated_input_tokens: int
    effective_budget: int
    mode: str


@dataclass(frozen=True)
class MinutesRun:
    minutes: MinutesOfMeeting
    segments: list[Segment]
    speaker_names: dict[str, str]
    mode: str
    chunk_count: int
    reduction_levels: int
    elapsed_seconds: float
    metrics: list[CallMetric]

    @property
    def provider_seconds(self) -> float:
        return sum(metric.elapsed_seconds for metric in self.metrics)


def _map_user(text: str, label: str, names: dict[str, str]) -> str:
    return (
        f"WINDOW: {label}\n"
        f"RESOLVED SPEAKERS: {json.dumps(names, ensure_ascii=False)}\n"
        f"DIARIZED TRANSCRIPT:\n{text}"
    )


def _reduce_user(
    partials: Sequence[PartialMinutes],
    names: dict[str, str] | None = None,
    roles: dict[str, str] | None = None,
) -> str:
    roster = ""
    if names is not None:
        roster = (
            f"SPEAKER NAMES: {json.dumps(names, ensure_ascii=False)}\n"
            f"SPEAKER ROLES: {json.dumps(roles or {}, ensure_ascii=False)}\n"
        )
    payload = [partial.model_dump(mode="json") for partial in partials]
    return roster + "PARTIAL MINUTES:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


class MinutesAgent:
    """Meeting-specific orchestration over an injected, reusable structured LLM."""

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        resolve_speaker_names: bool = True,
    ) -> None:
        self.llm = llm
        self.resolve_speaker_names = resolve_speaker_names
        # Names/roles for the meeting currently being processed. Set at the start
        # of run()/finalize() and read by the make_user/make_final_user closures
        # below, since MapReduceJob only knows about text in, model out.
        self._names: dict[str, str] = {}
        self._roles: dict[str, str] = {}

        self._job = MapReduceJob(
            llm,
            map_prompt=MOM_PROMPT,
            partial_schema=PartialMinutes,
            reduce_prompt=PARTIAL_REDUCE_PROMPT,
            final_prompt=FINAL_REDUCE_PROMPT,
            final_schema=MinutesOfMeeting,
            make_user=lambda text, label: _map_user(text, label, self._names),
            make_reduce_user=lambda partials: _reduce_user(partials),
            make_final_user=lambda partials: _reduce_user(partials, self._names, self._roles),
            task_prefix="mom",
        )

    @property
    def max_concurrent_calls(self) -> int:
        return self._job.max_concurrent_calls

    @property
    def effective_budget(self) -> int:
        return self._job.effective_budget

    def plan(self, transcript: Any) -> ExecutionPlan:
        segments = normalize_transcript(transcript)
        names = {speaker_id: speaker_id for speaker_id in speaker_ids(segments)}
        tokens, _ = self.llm.count_input_tokens(
            MOM_PROMPT,
            _map_user(render_segments(segments), "full_meeting", names),
            MinutesOfMeeting,
            exact=False,
        )
        return ExecutionPlan(
            segments=len(segments),
            estimated_input_tokens=tokens,
            effective_budget=self.effective_budget,
            mode="single-pass" if tokens <= self.effective_budget else "map-reduce",
        )

    def run(self, transcript: Any, *, resolve_names: bool | None = None) -> MinutesRun:
        segments = normalize_transcript(transcript)
        use_names = self.resolve_speaker_names if resolve_names is None else resolve_names
        self._names, self._roles = (
            self._resolve_speakers(segments) if use_names else self._fallback_roster(segments)
        )

        run = self._job.run(segments)
        minutes = self._ensure_participants(run.result, self._names, self._roles)

        return MinutesRun(
            minutes=minutes,
            segments=segments,
            speaker_names=self._names,
            mode=run.mode,
            chunk_count=run.chunk_count,
            reduction_levels=run.reduction_levels,
            elapsed_seconds=run.elapsed_seconds,
            metrics=run.metrics,
        )

    def run_file(self, path: str | Path, **kwargs: Any) -> MinutesRun:
        return self.run(load_transcript(path), **kwargs)


    def save(
        self, run: MinutesRun, output_dir: str | Path, basename: str = "mom"
    ) -> "MinutesArtifacts":
        from .rendering import save_minutes

        return save_minutes(run, output_dir, basename)

    # ---- meeting-specific helpers (not map-reduce mechanics) -------------

    def _fallback_roster(
        self, segments: Sequence[Segment]
    ) -> tuple[dict[str, str], dict[str, str]]:
        ids = speaker_ids(segments)
        return fallback_names(ids), {speaker_id: "" for speaker_id in ids}

    def _resolve_speakers(
        self, segments: Sequence[Segment]
    ) -> tuple[dict[str, str], dict[str, str]]:
        ids = speaker_ids(segments)
        ids_text = json.dumps(ids)

        def user(sample: Sequence[Segment]) -> str:
            return f"SPEAKER IDS: {ids_text}\nDIARIZED TRANSCRIPT:\n{render_segments(sample)}"

        sample = list(segments)
        tokens, _ = self.llm.count_input_tokens(SPEAKER_PROMPT, user(sample), SpeakerRoster)
        if tokens > self.effective_budget:
            first_turns = {
                speaker_id: next(item for item in segments if item.speaker == speaker_id)
                for speaker_id in ids
            }
            chosen = dict((item.index, item) for item in first_turns.values())
            for item in segments:
                candidate = [*chosen.values(), item]
                candidate = sorted({part.index: part for part in candidate}.values(), key=lambda part: part.index)
                estimate, _ = self.llm.count_input_tokens(
                    SPEAKER_PROMPT, user(candidate), SpeakerRoster, exact=False
                )
                if estimate <= self.effective_budget:
                    chosen = {part.index: part for part in candidate}
            sample = [chosen[index] for index in sorted(chosen)]

        roster = self.llm.structured(
            SPEAKER_PROMPT, user(sample), SpeakerRoster, task="mom_resolve_speakers"
        )
        returned = {speaker.speaker_id: speaker for speaker in roster.speakers}
        fallbacks = fallback_names(ids)
        names: dict[str, str] = {}
        roles: dict[str, str] = {}
        used: set[str] = set()
        for speaker_id in ids:
            identity = returned.get(speaker_id, SpeakerIdentity(speaker_id=speaker_id))
            name = identity.name.strip()
            if not name or name in used:
                name = fallbacks[speaker_id]
            names[speaker_id] = name
            roles[speaker_id] = identity.role.strip()
            used.add(name)
        return names, roles

    @staticmethod
    def _ensure_participants(
        minutes: MinutesOfMeeting,
        names: dict[str, str],
        roles: dict[str, str],
    ) -> MinutesOfMeeting:
        returned = {participant.speaker_id: participant for participant in minutes.participants}
        participants = [
            Participant(
                name=(returned.get(speaker_id).name if speaker_id in returned else name) or name,
                role=(
                    returned.get(speaker_id).role
                    if speaker_id in returned
                    else roles.get(speaker_id, "")
                ),
                speaker_id=speaker_id,
            )
            for speaker_id, name in names.items()
        ]
        return minutes.model_copy(update={"participants": participants})
