# Meeting Processing Service

A GPU-backed FastAPI service that converts meeting videos into:

- speaker-diarized transcripts;
- enrolled-speaker recognition in transcript segments when a voice profile matches;
- structured minutes of meeting (MOM); and
- structured action plans.

Long transcripts are processed with token-bounded map-reduce. Chunk boundaries
always preserve complete diarization segments: when adding a segment would
exceed the input budget, that entire segment starts the next chunk. Segments are
never split by words. If a single segment cannot fit by itself, processing stops
with a `TokenBudgetError` instead of silently truncating or splitting it.

## Architecture

```text
Client -> FastAPI -> Redis/RQ -> GPU worker
                                  |
Video -> WAV -> WhisperX -> diarized segments
                              |-> enrolled voice-profile matching
                              |-> action plan JSON
                              `-> meeting minutes JSON
```

The API accepts uploads and queues work. A single RQ worker loads the WhisperX
models once and processes jobs on one NVIDIA GPU.

## Requirements

- Python 3.11
- Redis
- FFmpeg
- An NVIDIA GPU with a compatible driver
- Docker, Docker Compose, and NVIDIA Container Toolkit (recommended)
- Access to the configured WhisperX and diarization models
- An OpenAI-compatible LLM endpoint that supports structured JSON output

## Configuration

Create a `.env` file in the project root:

```dotenv
GEMMA_API_KEY=your-api-key
GEMMA_BASE_URL=https://your-openai-compatible-endpoint/v1
```

When running outside Docker, Redis defaults to `redis://localhost:6379/0`. You
can override it with:

```dotenv
REDIS_URL=redis://localhost:6379/0
```

Non-secret settings are in [`config.py`](config.py), including:

- Whisper and diarization model paths;
- transcription batch size and compute type;
- LLM model and generation settings;
- input/output token limits;
- map-reduce safety factor, concurrency, and segment overlap.

With the current values, the map-reduce effective input budget is 17,000 tokens
(`20,000 * 0.85`). Prompt and schema tokens are part of that budget.

## Enrolled speaker recognition

The transcriber reads the shared `voice_profile/speakers.json` store. For each
diarization speaker, it compares embeddings from that speaker's longest speech
spans with enrolled profiles. A match above `voice_profile/config.py`'s
`THRESHOLD` adds `speaker_name`, `speaker_email`, and `speaker_match_score` to
that speaker's transcript segments. Diarization IDs remain present; speakers
without a confident profile match are returned without identity fields. The
SpeechBrain model is loaded lazily on CPU by the STT worker so it does not
compete with WhisperX and pyannote for GPU memory. With no enrolled profiles,
voice matching is skipped.

## Run with Docker Compose

Before starting, update the host model-cache path under `worker-gpu0.volumes` in
[`docker-compose.yml`](docker-compose.yml):

```yaml
- /your/local/huggingface/cache:/hub
```

Then build and start the API, Redis, and GPU worker:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`. Interactive API documentation
is available at `http://localhost:8000/docs`.

Only run one worker per GPU. The worker uses `SimpleWorker` because forking a
process after CUDA initialization can break GPU access.

## Run locally

Install FFmpeg and Redis, create a virtual environment, then install the Python
dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start Redis, then run the API and worker in separate terminals:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bash
python main.py
```

The local worker currently requires CUDA; CPU execution is not configured.

## API usage

### Submit a meeting

```bash
curl -X POST http://localhost:8000/meetings \
  -F "file=@/path/to/meeting.mp4"
```

Example response:

```json
{
  "job_id": "ee4c1095-4e0f-48fc-9703-52c74ac75a05",
  "status": "queued",
  "result": null,
  "error": null
}
```

### Check job status

```bash
curl http://localhost:8000/meetings/ee4c1095-4e0f-48fc-9703-52c74ac75a05
```

Job status is one of `queued`, `started`, `finished`, or `failed`. A finished
response contains:

```json
{
  "transcription": [],
  "action_plan": {"action_items": []},
  "minutes_of_meeting": {}
}
```

### Delete a job

```bash
curl -X DELETE http://localhost:8000/meetings/ee4c1095-4e0f-48fc-9703-52c74ac75a05
```

## Processing outputs

During processing, the service writes these files beside the uploaded video:

- `transcription_diarization.json`
- `action_plan.json`
- `mom.json`

After successful processing, the uploaded video and intermediate WAV file are
deleted. The API result remains available through the RQ job according to its
configured Redis retention period, or until the job is deleted.

## Project layout

```text
main.py                    FastAPI endpoints and RQ worker
pipeline.py                End-to-end meeting processing pipeline
stt_diarization.py         Audio extraction, transcription, and diarization
map_reduce.py              Shared token-aware map-reduce implementation
transcript.py              Transcript normalization and rendering
llms.py                    Structured LLM clients and token accounting
mom/                       Minutes schemas, prompts, agent, and rendering
action_plan/               Action-plan schemas, prompts, and agent
config.py                  Runtime and model settings
docker-compose.yml         API, Redis, and GPU-worker services
```
