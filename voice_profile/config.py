MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"
UPLOAD_DIR = BASE_DIR / "voices"
SPEAKER_STORE_PATH = BASE_DIR / "speakers.json"
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
MIN_AUDIO_SECONDS = 30
MAX_AUDIO_SECONDS = 120
DELETE_AFTER_PROCESSING = True
THRESHOLD = 0.3
ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
}
ACCEPTED_EXTENSIONS_FOR_DISPLAY = ", ".join(sorted(ALLOWED_EXTENSIONS))