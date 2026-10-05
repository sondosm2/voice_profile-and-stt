import argparse

from .speaker_identifier import SpeakerIdentifier
from .speaker_store import SpeakerStore
from loguru import logger

def enroll(name: str,email: str,audio: str,):
    logger.info("SPEAKER ENROLLMENT")
    logger.info(f"Name  : {name}")
    logger.info(f"Email : {email}")
    logger.info(f"Audio : {audio}") 
    identifier = SpeakerIdentifier()

    logger.info("\n[1] Creating speaker embedding...")

    embedding = identifier.get_embedding(audio)

    logger.info(
        f"[2] Embedding generated "
        f"(dimension={embedding.shape[0]})"
    )

    store = SpeakerStore()

    logger.info("[3] Saving speaker profile...")

    store.add_speaker(
        name=name,
        email=email,
        embedding=embedding,
    )

    logger.info("\nSpeaker enrolled successfully.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        required=True,
    )

    parser.add_argument(
        "--email",
        required=True,
    )

    parser.add_argument(
        "--audio",
        required=True,
    )

    args = parser.parse_args()

    enroll(
        name=args.name,
        email=args.email,
        audio=args.audio,
    )