"""Start the meeting STT RQ worker for the combined package."""

from .stt_service.endpoint.main import run_worker


if __name__ == "__main__":
    run_worker()