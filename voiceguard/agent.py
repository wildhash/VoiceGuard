"""VoiceGuard self-improving agent.

Implements the autonomous monitoring and self-improvement loop:

1. Execute an Airia agent pipeline to produce a voice-agent response.
2. Analyse the resulting audio with Modulate AI (Velma) to extract
   emotion scores, toxicity flags, and transcription.
3. Query Lightdash for aggregate performance metrics.
4. Feed the combined signals back into Airia's improvement pipeline so
   the agent can adjust its own behaviour.
"""

import logging
import os
from typing import Any, BinaryIO

import requests

from voiceguard.airia_client import AiriaClient
from voiceguard.lightdash import LightdashClient
from voiceguard.modulate import ModulateClient

logger = logging.getLogger(__name__)


METRICS_SQL = (
    "SELECT "
    "  AVG(satisfaction_score) AS avg_satisfaction, "
    "  AVG(anger_score)        AS avg_anger, "
    "  AVG(toxicity_score)     AS avg_toxicity, "
    "  COUNT(*)                AS total_interactions "
    "FROM agent_metrics "
    "WHERE date = CURRENT_DATE"
)


class VoiceGuardAgent:
    """Autonomous agent that monitors voice interactions and self-improves.

    Args:
        airia: Configured :class:`~voiceguard.airia_client.AiriaClient`.
        modulate: Configured :class:`~voiceguard.modulate.ModulateClient`.
        lightdash: Configured :class:`~voiceguard.lightdash.LightdashClient`.
        pipeline_id: UUID of the main Airia agent pipeline.
        improvement_pipeline_id: UUID of the self-improvement Airia pipeline.
    """

    def __init__(
        self,
        airia: AiriaClient | None = None,
        modulate: ModulateClient | None = None,
        lightdash: LightdashClient | None = None,
        pipeline_id: str | None = None,
        improvement_pipeline_id: str | None = None,
    ) -> None:
        self._airia = airia or AiriaClient()
        self._modulate = modulate or ModulateClient()
        self._lightdash = lightdash or LightdashClient()
        self._pipeline_id = pipeline_id or os.environ["AIRIA_PIPELINE_ID"]
        self._improvement_pipeline_id = (
            improvement_pipeline_id or os.environ["AIRIA_IMPROVEMENT_PIPELINE_ID"]
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        user_input: str,
        audio: BinaryIO | bytes | None = None,
    ) -> dict[str, Any]:
        """Execute one full monitoring and self-improvement cycle.

        Args:
            user_input: The query or task to send to the main agent pipeline.
            audio: Optional raw audio produced by the voice agent.  When
                provided it is analysed by Modulate AI.

        Returns:
            Dictionary with keys ``pipeline_result``, ``voice_analysis``
            (or ``None`` when no audio is provided), ``metrics``, and
            ``improvement_result``.
        """
        logger.info("Step 1 — executing main agent pipeline")
        pipeline_result = self._airia.execute_pipeline(
            pipeline_id=self._pipeline_id,
            user_input=user_input,
        )

        voice_analysis: dict[str, Any] | None = None
        if audio is not None:
            logger.info("Step 2 — analysing audio with Modulate AI (Velma)")
            voice_analysis = self._modulate.analyze(audio)

        logger.info("Step 3 — querying Lightdash for performance metrics")
        metrics = self._query_metrics()

        logger.info("Step 4 — feeding signals back into improvement pipeline")
        improvement_result = self._improve(voice_analysis, metrics)

        return {
            "pipeline_result": pipeline_result,
            "voice_analysis": voice_analysis,
            "metrics": metrics,
            "improvement_result": improvement_result,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query_metrics(self) -> dict[str, Any]:
        """Return today's aggregate performance metrics from Lightdash."""
        try:
            return self._lightdash.run_sql_query(METRICS_SQL)
        except (requests.RequestException, TimeoutError):
            logger.warning("Could not fetch Lightdash metrics", exc_info=True)
            return {}

    def _improve(
        self,
        voice_analysis: dict[str, Any] | None,
        metrics: dict[str, Any],
    ) -> Any:
        """Trigger the self-improvement pipeline with combined signals."""
        voice_summary = (
            f"Voice analysis: {voice_analysis}" if voice_analysis else "No voice analysis."
        )
        metrics_summary = f"Performance metrics: {metrics}" if metrics else "No metrics available."
        improvement_prompt = (
            f"{metrics_summary} {voice_summary} "
            "Based on these signals, review and adjust your detection criteria "
            "and response strategy to reduce negative emotion scores and toxicity."
        )
        return self._airia.execute_pipeline(
            pipeline_id=self._improvement_pipeline_id,
            user_input=improvement_prompt,
        )
