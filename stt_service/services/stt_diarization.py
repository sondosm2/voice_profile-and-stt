from dotenv import load_dotenv
load_dotenv()

import json
import os
import subprocess
import whisperx
from whisperx.diarize import DiarizationPipeline
from loguru import logger
from typing import List, Dict, Any, Optional
from itertools import groupby
import base64
from ..config import DEFAULT_WHISPER_MODEL_PATH, WHISPER_BATCH_SIZE, WHISPER_COMPUTE_TYPE, DEFAULT_WHISPERX_DIARIZATION_MODEL_PATH
import gc, torch

# ── 1. Audio Extraction ───────────────────────────────────────────────────────

def video_to_audio(video_path, output_audio_path):
    """Extract the audio track of a video or audio file as a 16 kHz mono WAV file."""
    logger.info(f"Extracting audio -> {output_audio_path}")
    # Uploads may already be .wav, in which case the output path is the input.
    if os.path.abspath(video_path) == os.path.abspath(output_audio_path):
        return output_audio_path

    # ffmpeg (unlike MoviePy's VideoFileClip) also handles audio-only inputs
    # such as .ogg/.mp3/.m4a voice notes.
    result = subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", output_audio_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"ffmpeg failed to extract audio: {result.stderr[-1000:]}")
        return None
    return output_audio_path

def audio_file_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return base64.b64encode(audio_bytes).decode("utf-8")

# ============================================================================
# Stage 3: Transcription + speaker diarization
# ============================================================================
 
class Transcriber:
    """Lazy-loaded whisperx transcription + alignment + diarization pipeline."""
 
    def __init__(self):
        # self.hf_token = DEFAULT_HF_TOKEN
        self.device = "cuda"
        self.batch_size = WHISPER_BATCH_SIZE
        self.compute_type = WHISPER_COMPUTE_TYPE
        self._model = None
        self._diarize_model = None
        self._voice_identifier = None

    def _match_enrolled_speakers(self, audio, segments):
        """Attach enrolled identities to diarized speaker IDs when confident."""
        from ...voice_profile.config import SPEAKER_STORE_PATH, THRESHOLD
        from ...voice_profile.services.speaker_store import SpeakerStore

        speakers = SpeakerStore(path=SPEAKER_STORE_PATH).get_speakers()
        if not speakers or not segments:
            return segments

        if self._voice_identifier is None:
            from ...voice_profile.services.speaker_identifier import SpeakerIdentifier

            # Keep the speaker model on CPU so it does not compete with WhisperX
            # and pyannote for the worker's GPU memory.
            self._voice_identifier = SpeakerIdentifier(device="cpu")

        # Build a voice sample per diarization cluster from its longest spans.
        spans = {}
        for segment in segments:
            speaker_id = segment.get("speaker_id")
            start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
            if speaker_id and end - start >= 1.5:
                spans.setdefault(speaker_id, []).append((start, end))

        identified = {}
        for speaker_id, speaker_spans in spans.items():
            embeddings = []
            # Several clean, long spans are more robust than embedding tiny turns.
            for start, end in sorted(speaker_spans, key=lambda span: span[1] - span[0], reverse=True)[:8]:
                waveform = torch.as_tensor(audio[int(start * 16000):int(end * 16000)])
                if waveform.numel() < int(1.5 * 16000):
                    continue
                embeddings.append(self._voice_identifier.get_embedding_from_waveform(waveform))
            if not embeddings:
                continue

            profile_embedding = torch.nn.functional.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=0)
            best = None
            for profile in speakers:
                stored = torch.tensor(profile["embedding"], dtype=torch.float32)
                score = self._voice_identifier.compare_embedding(profile_embedding, stored)
                if best is None or score > best["score"]:
                    best = {"name": profile["name"], "email": profile["email"], "score": score}
            if best and best["score"] >= THRESHOLD:
                identified[speaker_id] = best

        for segment in segments:
            match = identified.get(segment.get("speaker_id"))
            if match:
                segment["speaker_name"] = match["name"]
                segment["speaker_email"] = match["email"]
                segment["speaker_match_score"] = round(match["score"], 4)
        return segments
 
    def _load_whisper(self):
        if self._model is not None:
            logger.info("Whisper model already loaded.")
            return
        logger.info(f"Loading whisper model: {DEFAULT_WHISPER_MODEL_PATH}")
        self._model = whisperx.load_model(
            DEFAULT_WHISPER_MODEL_PATH, self.device, compute_type=self.compute_type
        )
        
 
    def _load_diarizer(self):
        if self._diarize_model is not None:
            logger.info("Diarization model already loaded.")
            return
        # if not self.hf_token:
        #     logger.warning(
        #         "No HF token provided -> diarization disabled, "
        #         "all speech will be assigned to SPEAKER_00."
        #     )
        #     return
        logger.info("Loading diarization model...")
        self._diarize_model = DiarizationPipeline(
            DEFAULT_WHISPERX_DIARIZATION_MODEL_PATH,
            # token=self.hf_token,
            device=self.device
            )
        

    @staticmethod

    def _flatten_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalizes and flattens WhisperX segments into speaker-labeled chunks.

        If speaker diarization information exists at the word level, this method 
        re-segments the input based on speaker transitions. In the absence of 
        word-level data, it retains the original segment boundaries.

        Args:
            segments: A list of raw WhisperX segment dictionaries.

        Returns:
            A flat list of dictionaries containing:
            {'start', 'end', 'text', 'speaker_id'}
        """
        flattened_segments = []

        for segment in segments:
            words = segment.get("words", [])

            # Fallback: Process original segment if word-level data is unavailable
            if not words:
                flattened_segments.append({
                    "start": round(float(segment.get("start", 0.0)), 3),
                    "end": round(float(segment.get("end", 0.0)), 3),
                    "text": segment.get("text", "").strip(),
                    "speaker_id": segment.get("speaker", "SPEAKER_00"),
                })
                continue

            # Group words by speaker to handle mid-segment speaker changes
            for speaker_id, group in groupby(words, key=lambda x: x.get("speaker", "SPEAKER_00")):
                word_list = [w for w in group if "start" in w and "end" in w]
                
                if not word_list:
                    continue

                flattened_segments.append({
                    "start": round(float(word_list[0]["start"]), 3),
                    "end": round(float(word_list[-1]["end"]), 3),
                    "text": " ".join(w["word"].strip() for w in word_list if w.get("word")),
                    "speaker_id": speaker_id,
                })

        return flattened_segments
    def transcribe(self, audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
        """
        Returns a flat list of segments:
            [{"start": float, "end": float, "text": str, "speaker_id": str}, ...]
        """ 
        self._load_whisper()
 
        audio = whisperx.load_audio(audio_path)
        result = self._model.transcribe(audio, batch_size=self.batch_size)
 
        # Word-level alignment
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, self.device,
            return_char_alignments=False,
        )
        del model_a 
        gc.collect()
        torch.cuda.empty_cache()
 
        # Speaker diarization (optional)
        self._load_diarizer()
        if self._diarize_model is not None:
            kwargs = {}
            if num_speakers:
                kwargs["min_speakers"] = num_speakers
                kwargs["max_speakers"] = num_speakers
            else:
                if min_speakers:
                    kwargs["min_speakers"] = min_speakers
                if max_speakers:
                    kwargs["max_speakers"] = max_speakers
 
            diarize_segments = self._diarize_model(audio, **kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
 
        segments = self._flatten_segments(result["segments"])
        return self._match_enrolled_speakers(audio, segments)
 


    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._diarize_model is not None:
            del self._diarize_model
            self._diarize_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Transcriber GPU memory freed.")

if __name__ == "__main__":
    video_path="/media/ai/96ba2917-db4a-4c43-9410-ee6f29403cde1/video_process/videos/YTDown_YouTube_Media_wYvab5hF25w_001_720p.mp4"
    output_audio="extracted_audio.wav"
    # Step 1: Extract audio from video
    audio_path = video_to_audio(video_path,output_audio)
    if audio_path is None:
        logger.error("Audio extraction failed. Exiting.")
        exit(1)

    # Step 2: Transcribe and diarize
    transcriber = Transcriber()
    segments = transcriber.transcribe(audio_path)
    with open("transcription_diarization.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
