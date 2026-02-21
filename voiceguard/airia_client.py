"""Airia pipeline execution client.

Wraps the ``airia`` Python SDK to provide a thin, testable interface for
executing agent pipelines.
"""

import os
from typing import Any

import airia as airia_sdk


class AiriaClient:
    """Client for executing Airia agent pipelines."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ["AIRIA_API_KEY"]
        self._client = airia_sdk.AiriaClient(api_key=key)

    def execute_pipeline(self, pipeline_id: str, user_input: str) -> Any:
        """Execute an Airia pipeline and return the result.

        Args:
            pipeline_id: UUID of the Airia pipeline to run.
            user_input: The prompt or payload to send to the pipeline.

        Returns:
            The pipeline execution result object from the Airia SDK.
        """
        return self._client.pipeline_execution.execute(
            pipeline_id=pipeline_id,
            user_input=user_input,
        )
