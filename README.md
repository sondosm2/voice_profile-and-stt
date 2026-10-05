# Meeting STT Bundle

This project combines meeting transcription with speaker voice profiling. First,
users enroll voice profiles for known speakers. Then, during speech-to-text
processing, the system matches each diarized speaker against that profile store
and labels the transcript with the correct speaker when the match is strong
enough.

## Project flow

1. Enroll a speaker voice profile.
2. Store the voice embedding and metadata in the profile database.
3. Run meeting transcription and diarization.
4. Compare each detected speaker to the enrolled profiles.
5. Return transcript segments with speaker names and confidence scores when
   there is a valid match.

This lets the STT service recognize who is speaking in a meeting instead of only
identifying anonymous diarized segments.

## Key services

- `stt_service`: transcription, diarization, and meeting pipeline
- `voice_profile`: speaker enrollment, identification, and profile storage

## Endpoints

- `GET /health`
- `POST /stt/meetings`
- `GET /stt/meetings/{job_id}`
- `DELETE /stt/meetings/{job_id}`
- `POST /voice/speakers/enroll`
- `POST /voice/speakers/identify`
- `GET /voice/speakers`
- `DELETE /voice/speakers/{email}`

When at least one voice profile is enrolled, the STT worker compares each
diarized speaker against the same profile store. Transcript segments keep their
diarization `speaker_id` and include `speaker_name`, `speaker_email`, and
`speaker_match_score` when a match meets the voice-profile threshold. Unknown
or low-confidence speakers retain only their diarization ID.

## Run

```bash
python -m uvicorn meeting_stt_bundle.main:app --host 0.0.0.0 --port 8000
python -m meeting_stt_bundle.worker
```

Install dependencies from both service requirement files. The meeting STT
service requires Redis, FFmpeg, model access, and its LLM environment values.

