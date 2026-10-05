import json
from pathlib import Path

import torch


class SpeakerStore:

    def __init__(self, path="speakers.json"):

        self.path = Path(path)

        if not self.path.exists():
            self._save([])

    # ---------------------------------------------------------
    # Internal save
    # ---------------------------------------------------------

    def _save(self, data):

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def _load(self):

        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------------------------------------
    # Add speaker
    # ---------------------------------------------------------

    def add_speaker(
        self,
        name: str,
        email: str,
        embedding: torch.Tensor,
    ):

        data = self._load()

        # Remove old profile with same email
        data = [
            speaker
            for speaker in data
            if speaker["email"] != email
        ]

        speaker = {
            "name": name,
            "email": email,
            "embedding": embedding.tolist(),
        }

        data.append(speaker)

        self._save(data)

    # ---------------------------------------------------------
    # Get all speakers
    # ---------------------------------------------------------

    def get_speakers(self):

        return self._load()

    def delete_speaker(self, email: str) -> bool:

        data = self._load()
        remaining = [speaker for speaker in data if speaker["email"] != email]

        if len(remaining) == len(data):
            return False

        self._save(remaining)
        return True