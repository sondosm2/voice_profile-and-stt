"""Small public API for Idraak's minutes post-processor.

The application entrypoint can live anywhere; it creates a provider and injects it::

    from idraak.video_analysis.postprocessing.llms import OpenAICompatibleLLM
    from idraak.video_analysis.postprocessing.mom import MinutesAgent

    llm = OpenAICompatibleLLM.from_gemma_env(max_input_tokens=16_000)
    agent = MinutesAgent(llm, max_concurrent_calls=3)
    run = agent.run_file(transcript_path)
    agent.save(run, output_dir=meeting_artifacts)  # mom.json + mom.md

Use ``GeminiLLM.from_env()`` instead to read ``GOOGLE_API_KEY``.
"""

from ..llms import (
    CallMetric,
    GeminiLLM,
    LLMConfigurationError,
    OpenAICompatibleLLM,
    StructuredLLM,
    StructuredOutputError,
    TokenBudgetError,
)
from .agent import (
    ExecutionPlan,
    MinutesAgent,
    MinutesRun,
    Segment,
    TranscriptError,
    load_transcript,
    normalize_transcript,
    render_segments,
)
from .models import (
    Decision,
    ImportantNote,
    Issue,
    MinutesOfMeeting,
    OpenQuestion,
    Participant,
    PartialMinutes,
    SpeakerIdentity,
    SpeakerRoster,
    Topic,
)
from .rendering import MinutesArtifacts, save_minutes, to_markdown

__all__ = [
    "CallMetric",
    "Decision",
    "ExecutionPlan",
    "GeminiLLM",
    "ImportantNote",
    "Issue",
    "LLMConfigurationError",
    "MinutesAgent",
    "MinutesArtifacts",
    "MinutesOfMeeting",
    "MinutesRun",
    "OpenAICompatibleLLM",
    "OpenQuestion",
    "Participant",
    "PartialMinutes",
    "Segment",
    "SpeakerIdentity",
    "SpeakerRoster",
    "StructuredLLM",
    "StructuredOutputError",
    "TokenBudgetError",
    "Topic",
    "TranscriptError",
    "load_transcript",
    "normalize_transcript",
    "render_segments",
    "save_minutes",
    "to_markdown",
]

