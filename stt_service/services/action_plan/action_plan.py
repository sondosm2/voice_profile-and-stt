"""Action item extraction, built the same way minutes.py is: a StructuredLLM
plus a MapReduceJob. No chunking/reduce code lives here — it's all in
mapreduce.py and shared with the minutes agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

from ..llms import CallMetric, StructuredLLM
from ..map_reduce import MapReduceJob
from ..transcript import Segment, load_transcript, normalize_transcript, render_segments


# --------------------------------------------------------------------------- #
# Schema — partial and final are the SAME schema: extracting one chunk and
# merging N chunks' worth of items are both just "give me a deduped list of
# action items", so there's only one model, not a Partial/Final pair.
# --------------------------------------------------------------------------- #

class ActionItem(BaseModel):
    action: str = Field(
        description=(
            "A single concrete, actionable task extracted from the meeting, phrased as "
            "an imperative (e.g. 'Send the updated budget to finance'). Do not include "
            "vague intentions, general discussion topics, or decisions with no follow-up task."
        )
    )
    owner: Optional[str] = Field(
        default=None,
        description=(
            "Name of the person responsible for completing the task, as stated or clearly "
            "implied in the transcript. Use null if no owner is explicitly identified."
        ),
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Deadline exactly as mentioned, normalized to ISO 8601 when unambiguous. Null if none.",
    )
    priority: Optional[str] = Field(
        default=None, description="One of 'high', 'medium', 'low', or null if it can't be inferred."
    )
    status: Optional[str] = Field(
        default=None,
        description="One of 'not_started', 'in_progress', 'blocked', 'completed', or null if unstated.",
    )
    evidence: Optional[str] = Field(
        default=None,
        description="A short, near-verbatim supporting quote (<=1 sentence) from the transcript.",
    )


class ActionPlan(BaseModel):
    action_items: list[ActionItem] = Field(
        description=(
            "All concrete action items in this transcript/window, one entry per distinct task. "
            "Deduplicate items referring to the same task. Empty list if nothing actionable — "
            "never fabricate items.You should write in the transcript's dominant language."
        )
    )


MAP_PROMPT = (
    "You extract concrete, actionable tasks from a meeting transcript window. "
    "Only include items that are clearly actionable — no general discussion or vague intentions. "
    "Write in the transcript's dominant language."
)

REDUCE_PROMPT = (
    "You are given several ActionPlan extractions taken from consecutive windows of the same "
    "meeting. Merge them into a single ActionPlan: keep every distinct task, and where two items "
    "clearly refer to the same task (same action, same owner/intent even if worded differently), "
    "merge them into one entry instead of listing it twice. Prefer the most specific owner/due_date/"
    "priority/status/evidence available across the duplicates. Invent nothing."
)


def _make_user(text: str, label: str) -> str:
    return f"WINDOW: {label}\nDIARIZED TRANSCRIPT:\n{text}"


def _make_reduce_user(partials: Sequence[ActionPlan]) -> str:
    import json

    payload = [p.model_dump(mode="json") for p in partials]
    return "PARTIAL ACTION PLANS:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ActionPlanRun:
    action_plan: ActionPlan
    segments: list[Segment]
    mode: str
    chunk_count: int
    reduction_levels: int
    elapsed_seconds: float
    metrics: list[CallMetric]


class ActionPlanAgent:
    """Meeting-specific config on top of the shared MapReduceJob."""

    def __init__(self, llm: StructuredLLM) -> None:
        self._job = MapReduceJob(
            llm,
            map_prompt=MAP_PROMPT,
            partial_schema=ActionPlan,
            reduce_prompt=REDUCE_PROMPT,
            final_prompt=REDUCE_PROMPT,   # merging-and-deduping IS the final step here
            final_schema=ActionPlan,
            make_user=_make_user,
            make_reduce_user=_make_reduce_user,
            task_prefix="action_plan",
        )

    def run(self, transcript: Any) -> ActionPlanRun:
        segments = normalize_transcript(transcript)
        run = self._job.run(segments)
        return ActionPlanRun(
            action_plan=run.result,
            segments=segments,
            mode=run.mode,
            chunk_count=run.chunk_count,
            reduction_levels=run.reduction_levels,
            elapsed_seconds=run.elapsed_seconds,
            metrics=run.metrics,
        )

    def run_file(self, path: str | Path) -> ActionPlanRun:
        return self.run(load_transcript(path))


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #

def main() -> None:
    import json
    from llms import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM.from_gemma_env()
    agent = ActionPlanAgent(llm)
    run = agent.run_file("transcription_diarization.json")

    result = run.action_plan.model_dump()
    with open("action_plan.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Action plan saved to: action_plan.json  ({run.mode})")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()