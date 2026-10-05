"""Generic token-bounded map-reduce over a StructuredLLM.

This is the part of the old ``MinutesAgent`` that had nothing to do with
meetings: chunk sizing, parallel "map" calls, and recursive "reduce" calls
until the merged result fits the model's input budget. Any agent (minutes,
action items, anything future) configures a ``MapReduceJob`` with its own
prompts/schemas and gets that behaviour for free — no copy-pasting the
chunking loop again.

Contract a caller supplies:
  map_prompt / partial_schema   -- extract one chunk (or, for single-pass,
                                    the whole transcript) into a "partial"
  reduce_prompt                 -- merge N partials into one partial
  final_prompt / final_schema   -- turn the last partial(s) into the real
                                    output type. For single-pass runs this
                                    is called directly on the full transcript
                                    (same as minutes.py's original behaviour:
                                    MOM_PROMPT + MinutesOfMeeting, one call,
                                    no separate partial step).
  make_user(text, label)             -> user message for map / single-pass
  make_reduce_user(partials)         -> user message for intermediate reduce
  make_final_user(partials)          -> user message for the final call
                                         (defaults to make_reduce_user; use
                                         this hook to inject extra context,
                                         e.g. a resolved speaker roster)

If partial_schema == final_schema and reduce_prompt == final_prompt (the
action-items case: it's just a list to merge and dedupe), pass the same
value for both — nothing forces them apart.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from pydantic import BaseModel

from .llms import CallMetric, StructuredLLM, TokenBudgetError
from .transcript import Segment, render_segments
from ..config import CHUNK_OVERLAP_SEGMENTS, MAX_CONCURRENT_CALLS, INPUT_SAFETY_FACTOR

R = TypeVar("R")
I = TypeVar("I")


def parallel_map(items: Sequence[I], fn: Callable[[int, I], R], max_concurrent_calls: int) -> list[R]:
    """Run at most N provider calls together and preserve input order."""
    if max_concurrent_calls < 1:
        raise ValueError("max_concurrent_calls must be at least 1")
    if max_concurrent_calls == 1 or len(items) < 2:
        return [fn(i, item) for i, item in enumerate(items)]
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(max_concurrent_calls, len(items))) as pool:
        futures = {pool.submit(fn, i, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [r for r in results if r is not None]


@dataclass(frozen=True)
class MapReduceRun:
    result: BaseModel
    mode: str
    chunk_count: int
    reduction_levels: int
    metrics: list[CallMetric]
    elapsed_seconds: float


class MapReduceJob:
    """A configured map-reduce extraction over a StructuredLLM."""

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        map_prompt: str,
        partial_schema: type[BaseModel],
        reduce_prompt: str,
        final_prompt: str,
        final_schema: type[BaseModel],
        make_user: Callable[[str, str], str],
        make_reduce_user: Callable[[Sequence[BaseModel]], str],
        make_final_user: Callable[[Sequence[BaseModel]], str] | None = None,
        task_prefix: str = "extract",
    ) -> None:
        if MAX_CONCURRENT_CALLS < 1:
            raise ValueError("max_concurrent_calls must be at least 1")
        if not 0 < INPUT_SAFETY_FACTOR <= 1:
            raise ValueError("input_safety_factor must be in (0, 1]")
        self.llm = llm
        self.map_prompt = map_prompt
        self.partial_schema = partial_schema
        self.reduce_prompt = reduce_prompt
        self.final_prompt = final_prompt
        self.final_schema = final_schema
        self.make_user = make_user
        self.make_reduce_user = make_reduce_user
        self.make_final_user = make_final_user or make_reduce_user
        self.task_prefix = task_prefix
        self.max_concurrent_calls = MAX_CONCURRENT_CALLS
        self.input_safety_factor = INPUT_SAFETY_FACTOR
        self.chunk_overlap_segments = max(0, CHUNK_OVERLAP_SEGMENTS)

    @property
    def effective_budget(self) -> int:
        return int(self.llm.max_input_tokens * self.input_safety_factor)

    # ---- whole-transcript entry point ------------------------------------

    def run(self, segments: Sequence[Segment]) -> MapReduceRun:
        started = time.perf_counter()
        metric_offset = len(self.llm.metrics)

        full_user = self.make_user(render_segments(segments), "full_meeting")
        full_tokens, _ = self.llm.count_input_tokens(self.map_prompt, full_user, self.final_schema)

        if full_tokens <= self.effective_budget:
            result = self.llm.structured(
                self.map_prompt, full_user, self.final_schema, task=f"{self.task_prefix}_full"
            )
            mode, chunks, levels = "single-pass", 1, 0
        else:
            planned = self._plan_chunks(segments)
            partials = parallel_map(
                planned,
                lambda i, chunk: self.extract(chunk, f"chunk_{i + 1:05d}"),
                self.max_concurrent_calls,
            )
            result, levels = self.merge(partials)
            chunks = len(planned)
            mode = (
                f"token-map-reduce ({chunks} chunks, {levels} reduce levels, "
                f"concurrency<={self.max_concurrent_calls})"
            )

        return MapReduceRun(
            result=result,
            mode=mode,
            chunk_count=chunks,
            reduction_levels=levels,
            metrics=list(self.llm.metrics[metric_offset:]),
            elapsed_seconds=time.perf_counter() - started,
        )

    # ---- map-only / reduce-only entry points, for streaming callers ------

    def extract(self, segments: Sequence[Segment], label: str) -> BaseModel:
        """Map one already-sized chunk into a partial."""
        return self.llm.structured(
            self.map_prompt, self.make_user(render_segments(segments), label), self.partial_schema,
            task=f"{self.task_prefix}_{label}",
        )

    def extract_window(self, segments: Sequence[Segment], label: str) -> list[BaseModel]:
        """Map a window that may itself need splitting (streaming ingestion)."""
        tokens, _ = self.llm.count_input_tokens(
            self.map_prompt, self.make_user(render_segments(segments), label), self.partial_schema
        )
        if tokens <= self.effective_budget:
            return [self.extract(segments, label)]
        chunks = self._plan_chunks(segments)
        return parallel_map(
            chunks,
            lambda i, chunk: self.extract(chunk, f"{label}_{i + 1:03d}"),
            self.max_concurrent_calls,
        )

    def merge(self, partials: Sequence[BaseModel]) -> tuple[BaseModel, int]:
        """Reduce N partials down to one final-schema result."""
        if not partials:
            raise ValueError("At least one partial extraction is required.")
        current = list(partials)
        level = 0
        while True:
            final_user = self.make_final_user(current)
            tokens, _ = self.llm.count_input_tokens(self.final_prompt, final_user, self.final_schema)
            if tokens <= self.effective_budget:
                return (
                    self.llm.structured(
                        self.final_prompt, final_user, self.final_schema,
                        task=f"{self.task_prefix}_reduce_final",
                    ),
                    level,
                )
            groups = self._reduction_groups(current)
            if all(len(group) == 1 for group in groups):
                raise TokenBudgetError("The reducer cannot combine two partials within max_input_tokens.")
            level += 1
            current = parallel_map(
                groups,
                lambda i, group: group[0] if len(group) == 1 else self.llm.structured(
                    self.reduce_prompt, self.make_reduce_user(group), self.partial_schema,
                    task=f"{self.task_prefix}_reduce_level_{level}_group_{i + 1}",
                ),
                self.max_concurrent_calls,
            )

    # ---- chunk sizing -----------------------------------------------------

    def _plan_chunks(self, segments: Sequence[Segment]) -> list[list[Segment]]:
        envelope_tokens, _ = self.llm.count_input_tokens(
            self.map_prompt, self.make_user("", "chunk_00000"), self.partial_schema, exact=False
        )
        payload_budget = self.effective_budget - envelope_tokens
        if payload_budget <= 0:
            raise TokenBudgetError("The prompt and output schema exhaust the input-token budget.")

        rough: list[list[Segment]] = []
        current: list[Segment] = []
        for seg in segments:
            candidate = [*current, seg]
            if current and self.llm.estimate_text_tokens(render_segments(candidate)) > payload_budget:
                # Keep transcript segments atomic.  The segment that crosses the
                # token budget starts the next chunk instead of being split into
                # word-based synthetic segments.
                rough.append(current)
                keep = min(self.chunk_overlap_segments, max(0, len(current) - 1))
                current = [*current[-keep:], seg] if keep else [seg]
            else:
                current = candidate
        if current:
            rough.append(current)

        ready: list[list[Segment]] = []
        queue = list(rough)
        while queue:
            chunk = queue.pop(0)
            label = f"chunk_{len(ready) + 1:05d}"
            tokens, _ = self.llm.count_input_tokens(
                self.map_prompt, self.make_user(render_segments(chunk), label), self.partial_schema
            )
            if tokens <= self.effective_budget:
                ready.append(chunk)
                continue
            if len(chunk) == 1:
                raise TokenBudgetError(
                    f"Segment {chunk[0].index} cannot fit in the effective input budget "
                    "without splitting the segment."
                )
            mid = len(chunk) // 2
            overlap = min(self.chunk_overlap_segments, max(0, mid - 1))
            queue[0:0] = [chunk[:mid], chunk[mid - overlap:]]
        return ready

    def _reduction_groups(self, partials: Sequence[BaseModel]) -> list[list[BaseModel]]:
        groups: list[list[BaseModel]] = []
        current: list[BaseModel] = []
        for partial in partials:
            candidate = [*current, partial]
            tokens, _ = self.llm.count_input_tokens(self.reduce_prompt, self.make_reduce_user(candidate), self.partial_schema)
            if current and tokens > self.effective_budget:
                groups.append(current)
                current = [partial]
            else:
                current = candidate
        if current:
            groups.append(current)
        return groups
