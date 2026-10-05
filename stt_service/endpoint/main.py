
import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from loguru import logger
from redis import Redis
from rq import Queue, SimpleWorker
from rq.job import Job as RQJob
from rq.exceptions import NoSuchJobError

from .pipeline import MeetingProcessor


# --------------------------------------------------------------------------
# Redis / Queue setup — shared by both the API process and the worker process.
# This is how the two separate processes talk to each other: the API writes
# a job into Redis, the worker picks it up and writes the result back.
# --------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(REDIS_URL)

# One queue, single GPU worker consuming it (see note in run_worker()).
meeting_queue = Queue("meetings", connection=redis_conn, default_timeout=3600)

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Job function — this is what actually runs inside the worker process.
# MeetingProcessor (and the GPU models it loads) is created ONCE per worker
# process via this lazy singleton, not once per job.
# --------------------------------------------------------------------------
_processor: MeetingProcessor | None = None


def get_processor() -> MeetingProcessor:
    global _processor
    if _processor is None:
        logger.info("Loading MeetingProcessor models into worker process...")
        _processor = MeetingProcessor()
        logger.info("Models loaded.")
    return _processor


def process_meeting_job(video_path: str) -> dict:
    """Entry point RQ calls for each job. Must stay a plain sync function."""
    import asyncio

    processor = get_processor()
    logger.info(f"Starting job for {video_path}")
    result = asyncio.run(processor.process(video_path))
    logger.info(f"Finished job for {video_path}")

    # Clean up uploaded video + intermediate audio once processing succeeds.
    try:
        Path(video_path).unlink(missing_ok=True)
        Path(video_path).with_suffix(".wav").unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Cleanup failed for {video_path}: {e}")

    return result


# --------------------------------------------------------------------------
# FastAPI app — run via `uvicorn meeting_service:app`
# --------------------------------------------------------------------------
app = FastAPI(title="Meeting Processing API")


class JobResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None


@app.post("/meetings", status_code=202, response_model=JobResponse)
async def submit_meeting(file: UploadFile = File(...)):
    job_id = str(uuid4())
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job = meeting_queue.enqueue(
        process_meeting_job,
        str(video_path),
        job_id=job_id,
        job_timeout=3600,
    )
    return JobResponse(job_id=job.id, status=job.get_status())


@app.get("/meetings/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    try:
        job = RQJob.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()  # queued | started | finished | failed
    response = JobResponse(job_id=job.id, status=status)

    if status == "finished":
        response.result = job.return_value()
    elif status == "failed":
        response.error = str(job.exc_info)[-1000:]

    return response


@app.delete("/meetings/{job_id}", status_code=204)
async def delete_job(job_id: str):
    try:
        job = RQJob.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail="Job not found")
    job.delete()


# --------------------------------------------------------------------------
# Worker entry point — run via `python meeting_service.py`
# --------------------------------------------------------------------------
def run_worker():
    # Warm up models before accepting jobs.
    get_processor()
    logger.info("Worker ready, listening on 'meetings' queue...")
    # SimpleWorker (no fork) — forking after CUDA is initialized breaks
    # GPU access in the child process. Run ONE of these per GPU.
    worker = SimpleWorker([meeting_queue], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    run_worker()