"""Pydantic contracts for the minutes agent.

These are the fields exposed to the rest of Idraak.  Provider code deliberately lives
outside this package so another post-processing agent can use the same LLM clients.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class Decision(Model):
    outline: str
    owner: str = ""
    owner_role: str = ""
    stakeholders: list[str] = Field(default_factory=list)


class OpenQuestion(Model):
    question: str
    asked_by: str
    answered_by: str = ""
    answer: str = ""
    status: Literal["answered", "partially_answered", "unanswered"]


class Topic(Model):
    title: str
    summary: str
    participants: list[str] = Field(default_factory=list)


class Issue(Model):
    title: str
    description: str
    raised_by: str = ""
    severity: Literal["low", "medium", "high"]
    status: Literal["open", "in_discussion", "resolved"]


class ImportantNote(Model):
    note: str
    priority: Literal["low", "medium", "high"]


class Participant(Model):
    name: str
    role: str = ""
    speaker_id: str = ""


class SpeakerIdentity(Model):
    """Internal name resolution result; intentionally no confidence/evidence fields."""

    speaker_id: str
    name: str = ""
    role: str = ""


class SpeakerRoster(Model):
    speakers: list[SpeakerIdentity] = Field(default_factory=list)


class PartialMinutes(Model):
    """The map result for one token-bounded transcript window."""

    output_language: str
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    important_notes: list[ImportantNote] = Field(default_factory=list)


class MinutesOfMeeting(Model):
    language: str
    title: str
    participants: list[Participant] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    important_notes: list[ImportantNote] = Field(default_factory=list)
    summary: str

