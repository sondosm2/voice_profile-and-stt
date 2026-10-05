from pathlib import Path
from typing import Optional

import torch
import torchaudio
import torch.nn.functional as F
from loguru import logger
from speechbrain.inference.speaker import SpeakerRecognition
from ..config import (
    ALLOWED_EXTENSIONS,
    MAX_AUDIO_SECONDS,
    MIN_AUDIO_SECONDS,
    MODEL_DIR,
    MODEL_SOURCE,
)


class SpeakerIdentifier:

    MODEL_SOURCE = MODEL_SOURCE

    ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Loading SpeechBrain speaker model on {self.device}..."
        )

        self.model = SpeakerRecognition.from_hparams(
            source=self.MODEL_SOURCE,
            savedir=model_dir,
            run_opts={"device": self.device},
        )

        logger.info("SpeechBrain speaker model loaded.")

    # ---------------------------------------------------------
    # Validate audio extension
    # ---------------------------------------------------------

    def validate_audio_extension(self, audio_path: str) -> bool:

        extension = Path(audio_path).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            logger.error(
                f"Unsupported audio format: '{extension}'. "
                f"Allowed formats: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
            )
            return False

        return True

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------

    def load_audio(
        self,
        audio_path: str,
        validate_duration: bool = True,
    ) -> Optional[torch.Tensor]:

        # Validate extension first
        if not self.validate_audio_extension(audio_path):
            return None

        # Check file exists
        if not Path(audio_path).is_file():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        try:
            signal, sample_rate = torchaudio.load(audio_path)

        except Exception as e:
            logger.error(
                f"Failed to load audio '{audio_path}': {e}"
            )
            return None

        duration = signal.shape[-1] / sample_rate
        if duration <= 0:
            logger.error("Audio file is empty")
            return None
        if (
            validate_duration
            and (duration < MIN_AUDIO_SECONDS or duration > MAX_AUDIO_SECONDS)
        ):
            logger.error(
                f"Audio duration must be between {MIN_AUDIO_SECONDS} and "
                f"{MAX_AUDIO_SECONDS} seconds; got {duration:.2f} seconds"
            )
            return None

        # Stereo -> mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        # SpeechBrain speaker model expects 16 kHz
        if sample_rate != 16000:
            logger.info(
                f"Resampling audio from {sample_rate} Hz to 16000 Hz"
            )

            signal = torchaudio.functional.resample(
                signal,
                sample_rate,
                16000,
            )

        # [1, samples] -> [samples]
        signal = signal.squeeze(0)

        return signal

    # ---------------------------------------------------------
    # Create speaker embedding
    # ---------------------------------------------------------

    def get_embedding(
        self,
        audio_path: str,
        validate_duration: bool = True,
    ) -> Optional[torch.Tensor]:

        waveform = self.load_audio(audio_path, validate_duration=validate_duration)

        if waveform is None:
            logger.error(
                f"Could not create embedding because audio "
                f"could not be loaded: {audio_path}"
            )
            return None

        return self.get_embedding_from_waveform(waveform)

    def get_embedding_from_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:

        waveform = waveform.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)

        embedding = embedding.squeeze()

        embedding = F.normalize(
            embedding,
            p=2,
            dim=0,
        )

        return embedding.cpu()

    # ---------------------------------------------------------
    # Compare two audio files
    # ---------------------------------------------------------

    def compare(
        self,
        audio1: str,
        audio2: str,
    ):

        emb1 = self.get_embedding(audio1)
        emb2 = self.get_embedding(audio2)

        if emb1 is None or emb2 is None:
            logger.error(
                "Cannot compare audio files because "
                "one or both embeddings could not be created."
            )
            return None

        score = F.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0),
            dim=1,
        ).item()

        return score

    # ---------------------------------------------------------
    # Compare embedding with stored embedding
    # ---------------------------------------------------------

    def compare_embedding(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ):

        embedding1 = F.normalize(
            embedding1,
            p=2,
            dim=0,
        )

        embedding2 = F.normalize(
            embedding2,
            p=2,
            dim=0,
        )

        score = F.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0),
            dim=1,
        ).item()

        return score