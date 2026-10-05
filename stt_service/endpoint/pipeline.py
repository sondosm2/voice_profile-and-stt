from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from ..services.stt_diarization import Transcriber, video_to_audio
from ..services.llms import OpenAICompatibleLLM
from ..services.action_plan.action_plan import ActionPlanAgent
from ..services.mom.agent import MinutesAgent


class MeetingProcessor:
    def __init__(self) -> None:
        self.transcriber = Transcriber()
        self.llm = OpenAICompatibleLLM.from_gemma_env()
        self.action_item_extractor = ActionPlanAgent(self.llm)
        self.minutes_agent = MinutesAgent(self.llm)

    def generate_action_plan(
        self,
        transcript: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        plan = self.action_item_extractor.run(transcript)

        action_plan = plan.action_plan.model_dump()

        output_path = output_dir / "action_plan.json"
        output_path.write_text(
            json.dumps(action_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Action plan generated: {} | mode={} | chunks={} | items={}",
            output_path,
            plan.mode,
            plan.chunk_count,
            len(action_plan["action_items"]),
        )

        return action_plan

    def generate_mom(
        self,
        transcript: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """
        Generate Minutes of Meeting directly from diarized transcript segments.
        """
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        run = self.minutes_agent.run(transcript)

        minutes = run.minutes.model_dump()

        output_path = output_dir / "mom.json"
        output_path.write_text(
            json.dumps(minutes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "MOM generated: {} | mode={} | chunks={} | levels={}",
            output_path,
            run.mode,
            run.chunk_count,
            run.reduction_levels,
        )

        return minutes

    async def process(self, video_path: str):
        """
        Complete meeting processing pipeline:
            video --> audio extraction --> transcription + diarization
                  --> minutes of meeting --> action plan --> structured result
        """
        video_path = Path(video_path).expanduser().resolve()

        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        output_dir = video_path.parent

        # ============================================================
        # 1. Extract audio
        # ============================================================
        audio_path = video_path.with_suffix(".wav")
        audio_path = video_to_audio(str(video_path), str(audio_path))

        if audio_path is None:
            logger.error("Audio extraction failed for {}", video_path)
            raise RuntimeError("Audio extraction failed.")

        logger.info("Audio extraction completed: {}", audio_path)

        # ============================================================
        # 2. Transcription + diarization
        # ============================================================
        transcript = self.transcriber.transcribe(str(audio_path))

        if not transcript:
            raise RuntimeError("Transcription returned no segments.")

        logger.info("Transcription completed: {} segments", len(transcript))

        transcription_path = output_dir / "transcription_diarization.json"
        transcription_path.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Transcription saved to {}", transcription_path)

        # ============================================================
        # 3. Minutes of Meeting + Action Plan (both consume the SAME
        #    raw transcript list — no pre-rendering to text)
        # ============================================================
        action_plan = self.generate_action_plan(transcript, output_dir=output_dir)

        minutes_of_meeting = self.generate_mom(transcript, output_dir=output_dir)

        # ============================================================
        # 4. Return structured result
        # ============================================================
        return {
            "transcription": transcript,
            "action_plan": action_plan,
            "minutes_of_meeting": minutes_of_meeting,
        }