# -- Transcription / diarization (whisperx) --
DEFAULT_WHISPER_MODEL_PATH = (
"Systran/faster-whisper-large-v3"
)

DEFAULT_WHISPERX_DIARIZATION_MODEL_PATH = ("pyannote/speaker-diarization-3.1")

WHISPER_BATCH_SIZE = 16
WHISPER_COMPUTE_TYPE = "float16"

# -- LLM (OpenAI-compatible endpoint, e.g. OpenRouter Gemma) --
# GEMMA_API_KEY / GEMMA_BASE_URL stay in .env since they're secrets; everything
# tunable lives here so there's one place to change the model or its behavior.
LLM_MODEL = "google/gemma-3-27b-it"
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.95
LLM_TIMEOUT_SECONDS = 180.0
LLM_MAX_INPUT_TOKENS = 20000
LLM_MAX_OUTPUT_TOKENS = 8000
LLM_ESTIMATED_BYTES_PER_TOKEN = 3.0
LLM_EXTRA_BODY = {
    "provider": {"require_parameters": False},
    "reasoning": {"effort": "none", "exclude": True},
    "top_k": 64,
}

MAX_CONCURRENT_CALLS = 1
INPUT_SAFETY_FACTOR = 0.85
CHUNK_OVERLAP_SEGMENTS = 2
