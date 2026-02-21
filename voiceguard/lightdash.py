"""Lightdash REST API client.

Supports SQL query execution (v2 async pattern), dashboard management,
and JWT-based dashboard embedding.
"""

import datetime
import os
import time
from typing import Any

import jwt
import requests


class LightdashClient:
    """Client for the Lightdash REST API (v1/v2)."""

    def __init__(
        self,
        access_token: str | None = None,
        instance_url: str | None = None,
        project_uuid: str | None = None,
    ) -> None:
        self._token = access_token or os.environ["LIGHTDASH_API_TOKEN"]
        self._base = (
            instance_url or os.getenv("LIGHTDASH_INSTANCE_URL", "https://app.lightdash.cloud")
        ).rstrip("/")
        self._project_uuid = project_uuid or os.environ["LIGHTDASH_PROJECT_UUID"]
        self._headers = {
            "Authorization": f"ApiKey {self._token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # SQL query execution (v2 — async pattern)
    # ------------------------------------------------------------------

    def run_sql_query(
        self,
        sql: str,
        limit: int = 500,
        max_wait_seconds: float = 30,
    ) -> dict[str, Any]:
        """Execute *sql* and return the full results dict.

        Uses the two-step v2 async pattern: submit the query, then fetch
        the results by queryUuid.

        Args:
            sql: SQL query string.
            limit: Maximum number of rows to return (default 500).
            max_wait_seconds: Maximum time to wait for results before raising
                :class:`TimeoutError`.

        Returns:
            The ``results`` value from the Lightdash response envelope.

        Raises:
            requests.HTTPError: On non-2xx HTTP responses.
            TimeoutError: If the query results are not ready within
                *max_wait_seconds*.
        """
        submit = requests.post(
            f"{self._base}/api/v2/projects/{self._project_uuid}/query/sql",
            headers=self._headers,
            json={"sql": sql, "limit": limit, "context": "api"},
            timeout=30,
        )
        submit.raise_for_status()
        query_uuid = submit.json()["results"]["queryUuid"]

        deadline = time.monotonic() + max_wait_seconds
        delay_seconds = 0.5

        while True:
            result = requests.get(
                f"{self._base}/api/v2/projects/{self._project_uuid}/query/{query_uuid}/results",
                headers=self._headers,
                timeout=30,
            )

            try:
                result.raise_for_status()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in {202, 404, 409}:
                    raise
            else:
                return result.json()["results"]

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Lightdash query results: {query_uuid}")

            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 5)

    # ------------------------------------------------------------------
    # Dashboard management
    # ------------------------------------------------------------------

    def list_dashboards(self) -> list[dict[str, Any]]:
        """Return a list of dashboards in the project."""
        resp = requests.get(
            f"{self._base}/api/v1/projects/{self._project_uuid}/dashboards",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def create_dashboard(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a new dashboard and return its details.

        Args:
            payload: Dashboard creation payload per the Lightdash API spec.

        Returns:
            Created dashboard object from the response.
        """
        resp = requests.post(
            f"{self._base}/api/v1/projects/{self._project_uuid}/dashboards",
            headers=self._headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def list_charts(self) -> list[dict[str, Any]]:
        """Return a list of saved charts in the project."""
        resp = requests.get(
            f"{self._base}/api/v1/projects/{self._project_uuid}/charts",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    # ------------------------------------------------------------------
    # JWT embedding
    # ------------------------------------------------------------------

    def get_embed_url(
        self,
        dashboard_uuid: str,
        embed_secret: str | None = None,
        ttl_hours: int = 1,
        user_attributes: dict[str, Any] | None = None,
    ) -> str:
        """Build a signed embed URL for the given dashboard.

        Args:
            dashboard_uuid: UUID of the dashboard to embed.
            embed_secret: HS256 signing secret (falls back to
                ``LIGHTDASH_EMBED_SECRET`` env var).
            ttl_hours: Token lifetime in hours (default 1).
            user_attributes: Optional per-viewer attribute map.

        Returns:
            Full iframe-ready embed URL string.
        """
        secret = embed_secret or os.environ["LIGHTDASH_EMBED_SECRET"]
        payload = {
            "content": {
                "type": "dashboard",
                "dashboardUuid": dashboard_uuid,
                "dashboardFiltersInteractivity": {"enabled": "all"},
                "canExportCsv": True,
                "canDateZoom": True,
            },
            "userAttributes": user_attributes or {},
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=ttl_hours),
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        return (
            f"{self._base}/embed/{self._project_uuid}"
            f"/dashboard/{dashboard_uuid}#{token}"
        )
