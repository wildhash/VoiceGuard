"""VoiceGuard entry point.

Loads environment variables from a ``.env`` file (if present) and runs a
single monitoring cycle with a sample prompt.
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from voiceguard import VoiceGuardAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="Handle the following customer support call and ensure compliance.",
        help="Text prompt to send to the main Airia pipeline",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional path to a WAV file (or other supported audio) to analyze with Modulate",
    )
    args = parser.parse_args()

    agent = VoiceGuardAgent()

    audio_bytes = None
    if args.audio:
        try:
            if not args.audio.is_file():
                logging.warning("Audio path is not a file: %s", args.audio)
            else:
                audio_bytes = args.audio.read_bytes()
        except OSError as exc:
            logging.warning("Could not read audio file %s: %s", args.audio, exc)

    result = agent.run_cycle(
        user_input=args.prompt,
        audio=audio_bytes,
    )

    def _to_output(value: object) -> object:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                return getattr(value, "output", value)
        if hasattr(value, "dict") and callable(getattr(value, "dict")):
            try:
                return value.dict()
            except Exception:
                return getattr(value, "output", value)
        return getattr(value, "output", value)

    print(
        json.dumps(
            {
                "pipeline_output": _to_output(result["pipeline_result"]),
                "voice_analysis": result["voice_analysis"],
                "metrics": result["metrics"],
                "improvement_output": _to_output(result["improvement_result"]),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
