"""Unit tests for VoiceGuard modules (all external HTTP calls are mocked)."""

import datetime
import io
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from voiceguard.agent import VoiceGuardAgent
from voiceguard.airia_client import AiriaClient
from voiceguard.lightdash import LightdashClient
from voiceguard.modulate import ModulateClient


# ---------------------------------------------------------------------------
# ModulateClient
# ---------------------------------------------------------------------------


class TestModulateClient:
    def _mock_response(self, data: dict) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = data
        resp.raise_for_status.return_value = None
        return resp

    def test_analyze_with_bytes(self):
        expected = {"transcription": "hello", "emotion": {"anger": 0.1}}
        with patch("voiceguard.modulate.requests.post") as mock_post:
            mock_post.return_value = self._mock_response(expected)
            client = ModulateClient(upload_endpoint="https://example.com/upload")
            result = client.analyze(b"fake-audio-data")

        assert result == expected
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "audio" in kwargs["files"]

    def test_analyze_with_file_like(self):
        expected = {"transcription": "test", "toxicity": 0.05}
        with patch("voiceguard.modulate.requests.post") as mock_post:
            mock_post.return_value = self._mock_response(expected)
            client = ModulateClient(upload_endpoint="https://example.com/upload")
            result = client.analyze(io.BytesIO(b"fake-audio"), filename="test.wav")

        assert result == expected

    def test_analyze_raises_on_http_error(self):
        with patch("voiceguard.modulate.requests.post") as mock_post:
            resp = MagicMock()
            resp.raise_for_status.side_effect = Exception("HTTP 500")
            mock_post.return_value = resp
            client = ModulateClient(upload_endpoint="https://example.com/upload")
            with pytest.raises(Exception, match="HTTP 500"):
                client.analyze(b"audio")


# ---------------------------------------------------------------------------
# LightdashClient
# ---------------------------------------------------------------------------


class TestLightdashClient:
    def _client(self):
        return LightdashClient(
            access_token="tok",
            instance_url="https://ld.example.com",
            project_uuid="proj-uuid",
        )

    def _mock_response(self, data: dict) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = data
        resp.raise_for_status.return_value = None
        return resp

    def test_run_sql_query(self):
        submit_resp = self._mock_response({"results": {"queryUuid": "q-uuid"}})
        result_resp = self._mock_response({"results": {"rows": [{"avg_satisfaction": 4.5}]}})

        with patch("voiceguard.lightdash.requests.post", return_value=submit_resp), \
             patch("voiceguard.lightdash.requests.get", return_value=result_resp):
            client = self._client()
            result = client.run_sql_query("SELECT 1")

        assert result == {"rows": [{"avg_satisfaction": 4.5}]}

    def test_list_dashboards(self):
        dashboards = [{"uuid": "d1", "name": "Agent Dashboard"}]
        resp = self._mock_response({"results": dashboards})
        with patch("voiceguard.lightdash.requests.get", return_value=resp):
            client = self._client()
            result = client.list_dashboards()
        assert result == dashboards

    def test_list_charts(self):
        charts = [{"uuid": "c1", "name": "Emotion Trends"}]
        resp = self._mock_response({"results": charts})
        with patch("voiceguard.lightdash.requests.get", return_value=resp):
            client = self._client()
            result = client.list_charts()
        assert result == charts

    def test_create_dashboard(self):
        created = {"uuid": "d2", "name": "New Dashboard"}
        resp = self._mock_response({"results": created})
        with patch("voiceguard.lightdash.requests.post", return_value=resp):
            client = self._client()
            result = client.create_dashboard({"name": "New Dashboard"})
        assert result == created

    def test_get_embed_url(self):
        import jwt as pyjwt

        secret = "embed-secret"
        client = self._client()
        url = client.get_embed_url("dash-uuid", embed_secret=secret)

        assert url.startswith("https://ld.example.com/embed/proj-uuid/dashboard/dash-uuid#")
        token_str = url.split("#", 1)[1]
        decoded = pyjwt.decode(token_str, secret, algorithms=["HS256"])
        assert decoded["content"]["dashboardUuid"] == "dash-uuid"
        assert decoded["content"]["type"] == "dashboard"


# ---------------------------------------------------------------------------
# AiriaClient
# ---------------------------------------------------------------------------


class TestAiriaClient:
    def test_execute_pipeline(self):
        fake_result = MagicMock()
        fake_result.output = "Agent response"

        mock_sdk = MagicMock()
        mock_sdk.pipeline_execution.execute.return_value = fake_result

        with patch("voiceguard.airia_client.airia_sdk.AiriaClient", return_value=mock_sdk):
            client = AiriaClient(api_key="ak-test-key")
            result = client.execute_pipeline("pipeline-uuid", "Hello agent")

        assert result.output == "Agent response"
        mock_sdk.pipeline_execution.execute.assert_called_once_with(
            pipeline_id="pipeline-uuid",
            user_input="Hello agent",
        )


# ---------------------------------------------------------------------------
# VoiceGuardAgent
# ---------------------------------------------------------------------------


class TestVoiceGuardAgent:
    def _make_agent(self):
        airia = MagicMock(spec=AiriaClient)
        airia.execute_pipeline.return_value = MagicMock(output="agent output")

        modulate = MagicMock(spec=ModulateClient)
        modulate.analyze.return_value = {"transcription": "hi", "emotion": {"anger": 0.2}}

        lightdash = MagicMock(spec=LightdashClient)
        lightdash.run_sql_query.return_value = {"rows": [{"avg_satisfaction": 4.0}]}

        agent = VoiceGuardAgent(
            airia=airia,
            modulate=modulate,
            lightdash=lightdash,
            pipeline_id="main-pipeline",
            improvement_pipeline_id="improve-pipeline",
        )
        return agent, airia, modulate, lightdash

    def test_run_cycle_without_audio(self):
        agent, airia, modulate, lightdash = self._make_agent()
        result = agent.run_cycle("Help this customer")

        assert result["pipeline_result"] is not None
        assert result["voice_analysis"] is None
        assert "rows" in result["metrics"]
        assert result["improvement_result"] is not None
        modulate.analyze.assert_not_called()
        assert airia.execute_pipeline.call_count == 2

    def test_run_cycle_with_audio(self):
        agent, airia, modulate, lightdash = self._make_agent()
        result = agent.run_cycle("Help this customer", audio=b"fake-audio")

        assert result["voice_analysis"] == {"transcription": "hi", "emotion": {"anger": 0.2}}
        modulate.analyze.assert_called_once_with(b"fake-audio")

    def test_run_cycle_lightdash_failure_is_graceful(self):
        agent, airia, modulate, lightdash = self._make_agent()
        lightdash.run_sql_query.side_effect = requests.RequestException("connection error")

        result = agent.run_cycle("test input")
        assert result["metrics"] == {}

    def test_improvement_prompt_includes_signals(self):
        agent, airia, modulate, lightdash = self._make_agent()
        agent.run_cycle("test", audio=b"audio-bytes")

        # The second execute_pipeline call is the improvement step
        calls = airia.execute_pipeline.call_args_list
        improvement_call = calls[1]
        prompt = improvement_call[1]["user_input"]
        assert "metrics" in prompt.lower() or "performance" in prompt.lower()
        assert "voice" in prompt.lower() or "analysis" in prompt.lower()
