"""Modulate AI (Velma) client for voice analysis.

Velma processes audio and returns emotion scores, toxicity flags,
transcription, and behavioral analysis.  The public API is not yet
launched; the preview upload endpoint is used here.  Discover the actual
endpoint URL by inspecting the Network tab in Chrome DevTools while
uploading a file at https://preview.modulate.ai/upload.html.
"""

import os
from typing import Any, BinaryIO

import requests


class ModulateClient:
    """Client for the Modulate AI (Velma) preview upload endpoint."""

    DEFAULT_ENDPOINT = "https://preview.modulate.ai/api/upload"

    def __init__(self, upload_endpoint: str | None = None) -> None:
        self._endpoint = upload_endpoint or os.getenv(
            "MODULATE_UPLOAD_ENDPOINT", self.DEFAULT_ENDPOINT
        )

    def analyze(self, audio: BinaryIO | bytes, filename: str = "audio.wav") -> dict[str, Any]:
        """Upload *audio* to the Velma preview endpoint and return the analysis.

        The preview service is unauthenticated and processes audio ephemerally.

        Args:
            audio: A file-like object opened in binary mode, or raw bytes.
            filename: Filename hint sent with the multipart upload.

        Returns:
            Parsed JSON response containing transcription, emotion scores,
            toxicity flags, and other analysis results.

        Raises:
            requests.HTTPError: If the server returns a non-2xx status.
        """
        if isinstance(audio, bytes):
            files: dict[str, Any] = {"audio": (filename, audio)}
        else:
            files = {"audio": (filename, audio)}

        response = requests.post(self._endpoint, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
