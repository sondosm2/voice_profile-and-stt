"""Combined API for meeting STT and voice-profile operations."""

from fastapi import FastAPI

from .stt_service.endpoint.main import app as stt_app
from .voice_profile.endpoint.main import app as voice_app
from .voice_profile.endpoint.main import load_model as load_voice_profile_model

app = FastAPI(title="Meeting STT and Voice Profile API")
app.mount("/stt", stt_app)
app.mount("/voice", voice_app)


@app.on_event("startup")
def initialize_mounted_services() -> None:
    load_voice_profile_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}