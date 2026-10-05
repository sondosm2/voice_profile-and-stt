"""
Speaker enrollment / identification API.

Endpoints:
    POST /speakers/enroll     -> name, email, audio file
    POST /speakers/identify   -> audio file
    GET  /speakers            -> list enrolled speakers (no embeddings)
    DELETE /speakers/{email}  -> remove a speaker

All tunable values (model, storage paths, allowed extensions, match
threshold) live in config.yaml, loaded via config.py.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8900
"""
import re
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..config import (
    ACCEPTED_EXTENSIONS_FOR_DISPLAY,
    ALLOWED_EXTENSIONS,
    DELETE_AFTER_PROCESSING,
    MAX_AUDIO_SECONDS,
    MAX_FILE_SIZE_BYTES,
    MIN_AUDIO_SECONDS,
    SPEAKER_STORE_PATH,
    THRESHOLD,
    UPLOAD_DIR,
)
from ..services.speaker_identifier import SpeakerIdentifier
from ..services.speaker_store import SpeakerStore

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Loaded once at startup, shared across requests (model load is expensive).
identifier: SpeakerIdentifier = None
store: SpeakerStore = None

app = FastAPI(title="Speaker Enrollment & Identification API")


@app.on_event("startup")
def load_model():
    global identifier, store
    if identifier is not None and store is not None:
        return
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    identifier = SpeakerIdentifier()
    store = SpeakerStore(path=SPEAKER_STORE_PATH)


# ---------------------------------------------------------------
# Response models
# ---------------------------------------------------------------
class EnrollResponse(BaseModel):
    status: str
    name: str
    email: str
    embedding_dim: int


class SpeakerScore(BaseModel):
    name: str
    email: str
    score: float


class IdentifyResponse(BaseModel):
    identified: bool
    threshold: float
    best_match: SpeakerScore | None
    # all_scores: list[SpeakerScore]


class SpeakerSummary(BaseModel):
    name: str
    email: str


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _validate_and_save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="Uploaded file has no extension.")

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio extension '{suffix}'. "
                f"Allowed: {ACCEPTED_EXTENSIONS_FOR_DISPLAY}"
            ),
        )

    dest = Path(UPLOAD_DIR) / f"{uuid.uuid4().hex}{suffix}"

    size = 0
    with open(dest, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"File exceeds max size of "
                        f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
                    ),
                )
            out.write(chunk)

    return dest


def _cleanup(path: Path):
    if DELETE_AFTER_PROCESSING:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.post("/speakers/enroll", response_model=EnrollResponse)
def enroll_speaker(
    name: str = Form(...),
    email: str = Form(...),
    audio: UploadFile = File(...),
):
    name = name.strip()
    email = email.strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Invalid email format.")

    audio_path = _validate_and_save_upload(audio)
    try:
        embedding = identifier.get_embedding(str(audio_path))
        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Audio must be between {MIN_AUDIO_SECONDS} and "
                    f"{MAX_AUDIO_SECONDS} seconds and use a supported format."
                ),
            )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"Could not process audio: {exc}")
    finally:
        _cleanup(audio_path)

    store.add_speaker(name=name, email=email, embedding=embedding)

    return EnrollResponse(
        status="enrolled",
        name=name,
        email=email,
        embedding_dim=embedding.shape[0],
    )


@app.post("/speakers/identify", response_model=IdentifyResponse)
def identify_speaker(audio: UploadFile = File(...)):
    speakers = store.get_speakers()
    if not speakers:
        raise HTTPException(status_code=404, detail="No enrolled speakers.")

    audio_path = _validate_and_save_upload(audio)
    try:
        test_embedding = identifier.get_embedding(
            str(audio_path), validate_duration=False
        )
        if test_embedding is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Audio must be a valid, non-empty recording and use a "
                    "supported format."
                ),
            )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"Could not process audio: {exc}")
    finally:
        _cleanup(audio_path)

    results = []
    for speaker in speakers:
        stored_embedding = torch.tensor(speaker["embedding"], dtype=torch.float32)
        score = identifier.compare_embedding(test_embedding, stored_embedding)
        results.append(
            SpeakerScore(name=speaker["name"], email=speaker["email"], score=score)
        )

    results.sort(key=lambda r: r.score, reverse=True)
    best = results[0]
    identified = best.score >= THRESHOLD

    return IdentifyResponse(
        identified=identified,
        threshold=THRESHOLD,
        best_match=best if identified else None,
        # all_scores=results,
    )


@app.get("/speakers", response_model=list[SpeakerSummary])
def list_speakers():
    return [
        SpeakerSummary(name=s["name"], email=s["email"]) for s in store.get_speakers()
    ]


@app.delete("/speakers/{email}")
def delete_speaker(email: str):
    deleted = store.delete_speaker(email.strip().lower())
    if not deleted:
        raise HTTPException(status_code=404, detail="Speaker not found.")
    return JSONResponse({"status": "deleted", "email": email})