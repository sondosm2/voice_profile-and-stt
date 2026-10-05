"""Deterministic JSON/Markdown output; no model calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .models import MinutesOfMeeting

if TYPE_CHECKING:
    from .agent import MinutesRun, Segment


@dataclass(frozen=True)
class MinutesArtifacts:
    json_path: Path
    markdown_path: Path


def _duration(segments: Sequence["Segment"]) -> str:
    ends = [segment.end for segment in segments if segment.end is not None]
    if not ends:
        return "Unknown"
    total = max(0, int(round(max(ends))))
    hours, remainder = divmod(total, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def to_markdown(minutes: MinutesOfMeeting, segments: Sequence["Segment"]) -> str:
    lines = [
        f"# {minutes.title}",
        "",
        f"**Duration:** {_duration(segments)}  |  **Language:** {minutes.language}",
        "",
        "## Participants",
        "",
    ]
    lines.extend(
        f"- **{person.name}**"
        + (f" — {person.role}" if person.role else "")
        + (f" `({person.speaker_id})`" if person.speaker_id else "")
        for person in minutes.participants
    )
    if not minutes.participants:
        lines.append("_None identified._")

    lines.extend(["", "## Summary", "", minutes.summary, "", "## Topics", ""])
    for index, topic in enumerate(minutes.topics, 1):
        lines.extend([f"### {index}. {topic.title}", "", topic.summary])
        if topic.participants:
            lines.extend(["", f"_Participants: {', '.join(topic.participants)}_"])
        lines.append("")
    if not minutes.topics:
        lines.extend(["_No topics extracted._", ""])

    lines.extend(["## Decisions", ""])
    for decision in minutes.decisions:
        owner = decision.owner or "—"
        if decision.owner_role:
            owner += f" ({decision.owner_role})"
        lines.append(f"- **{owner}:** {decision.outline}")
        if decision.stakeholders:
            lines.append(f"  - _Stakeholders: {', '.join(decision.stakeholders)}_")
    if not minutes.decisions:
        lines.append("_No decisions recorded._")

    icons = {"answered": "✅", "partially_answered": "🟡", "unanswered": "❓"}
    lines.extend(["", "## Open Questions", ""])
    for question in minutes.open_questions:
        lines.append(
            f"- {icons[question.status]} **{question.asked_by or '—'}:** {question.question}"
        )
        if question.answer:
            lines.append(f"  - **{question.answered_by or '—'}:** {question.answer}")
    if not minutes.open_questions:
        lines.append("_No questions raised._")

    lines.extend(["", "## Issues", ""])
    for issue in minutes.issues:
        raised_by = f" · raised by {issue.raised_by}" if issue.raised_by else ""
        lines.extend(
            [
                f"- **{issue.title}** `{issue.severity}` · `{issue.status}`{raised_by}",
                f"  - {issue.description}",
            ]
        )
    if not minutes.issues:
        lines.append("_No issues raised._")

    lines.extend(["", "## Important Notes", ""])
    lines.extend(f"- `{note.priority}` {note.note}" for note in minutes.important_notes)
    if not minutes.important_notes:
        lines.append("_None._")
    return "\n".join(lines).rstrip() + "\n"


def save_minutes(
    run: "MinutesRun", output_dir: str | Path, basename: str = "mom"
) -> MinutesArtifacts:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{basename}.json"
    markdown_path = directory / f"{basename}.md"
    json_path.write_text(run.minutes.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(to_markdown(run.minutes, run.segments), encoding="utf-8")
    return MinutesArtifacts(json_path=json_path, markdown_path=markdown_path)

