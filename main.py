"""VoiceGuard entry point.

Loads environment variables from a ``.env`` file (if present) and runs a
single monitoring cycle with a sample prompt.
"""

import logging

from dotenv import load_dotenv

from voiceguard import VoiceGuardAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

load_dotenv()


def main() -> None:
    agent = VoiceGuardAgent()
    result = agent.run_cycle(
        user_input="Handle the following customer support call and ensure compliance.",
    )
    print("Pipeline result  :", result["pipeline_result"])
    print("Voice analysis   :", result["voice_analysis"])
    print("Metrics          :", result["metrics"])
    print("Improvement      :", result["improvement_result"])


if __name__ == "__main__":
    main()
