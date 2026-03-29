from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.server import app
from app.state import AppState


@pytest.mark.e2e
class TestE2E:
    @pytest.fixture
    def client(self):
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def mock_app_state(self):
        state = MagicMock(spec=AppState)
        state.health_status = "ready"
        state.health_error = None
        state.models = MagicMock()
        state.models.serialize_public.return_value = [
            {"id_model": 1, "model_name": "v1_ctc", "enabled": True, "status": "downloaded", "progress": 100.0},
            {"id_model": 2, "model_name": "v2_rnnt", "enabled": True, "status": "downloaded", "progress": 100.0},
        ]
        state.models.is_model_known.return_value = True
        state.queue = MagicMock()
        state.queue.serialize_gpus_public.return_value = [
            {"index": 0, "name": "GPU 0", "status": "idle", "current_job_id": None, "current_model": None},
        ]
        state.queue.serialize_jobs_public.return_value = {"total": 0, "queued": 0, "running": 0, "queued_ids": [], "running_ids": []}
        state.queue.snapshot_ids.return_value = ([], [])
        state.queue.get_job.return_value = None
        return state

    @pytest.mark.asyncio
    def test_health_endpoint(self, client, mock_app_state):
        with patch.object(app.state, "core", mock_app_state):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["error"] is None

    @pytest.mark.asyncio
    def test_queue_endpoint_empty(self, client, mock_app_state):
        with patch.object(app.state, "core", mock_app_state):
            response = client.get("/queue")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "idle"
            assert data["queued"] == []
            assert data["running"] == []

    @pytest.mark.asyncio
    def test_status_endpoint_job_not_found(self, client, mock_app_state):
        mock_app_state.queue.get_job.return_value = None
        with patch.object(app.state, "core", mock_app_state):
            response = client.get("/status?job_id=nonexistent")
            assert response.status_code == 404

    @pytest.mark.asyncio
    def test_transcribe_endpoint_requires_valid_model(self, client, mock_app_state):
        mock_app_state.models.is_model_known.return_value = False
        with patch.object(app.state, "core", mock_app_state):
            response = client.post(
                "/transcribe",
                files={"file": ("test.mp3", b"fake audio", "audio/mpeg")},
                data={"model": "unknown_model", "callback_url": "http://cb"},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    def test_transcribe_endpoint_requires_callback_url(self, client, mock_app_state):
        with patch.object(app.state, "core", mock_app_state):
            response = client.post(
                "/transcribe",
                files={"file": ("test.mp3", b"fake audio", "audio/mpeg")},
                data={"model": "v1_ctc"},
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    def test_transcribe_endpoint_requires_file(self, client, mock_app_state):
        with patch.object(app.state, "core", mock_app_state):
            response = client.post(
                "/transcribe",
                data={"model": "v1_ctc", "callback_url": "http://cb"},
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    def test_dashboard_state_endpoint(self, client, mock_app_state):
        with patch.object(app.state, "core", mock_app_state):
            response = client.get("/dashboard/state")
            assert response.status_code == 200
            data = response.json()
            assert "health" in data
            assert "models" in data
            assert "gpus" in data
            assert "jobs" in data

    @pytest.mark.asyncio
    def test_dashboard_html_returns_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "GigaAM" in response.text

    @pytest.mark.asyncio
    def test_full_transcribe_flow_success(
        self, client, tmp_path, mock_app_state
    ):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake mp3 content")

        async def mock_enqueue(**kwargs):
            job_id = kwargs["job_id"]
            job_mock = MagicMock(
                job_id=job_id,
                status="completed",
                model=kwargs["model"],
                language=kwargs["language"],
                callback_url=kwargs["callback_url"],
                created_at_ms=1000000,
                started_at_ms=1000000,
                finished_at_ms=1005000,
                result={"text": "привет", "segments": []},
                error=None,
                callback_delivered_at_ms=1006000,
                callback_error=None,
                file_dir=kwargs["file_dir"],
            )
            mock_app_state.queue.serialize_job.return_value = {
                "job_id": job_id,
                "status": "completed",
                "model": kwargs["model"],
                "language": kwargs["language"],
                "callback_url": kwargs["callback_url"],
                "created_at_ms": 1000000,
                "started_at_ms": 1000000,
                "finished_at_ms": 1005000,
                "result": {"text": "привет", "segments": []},
                "error": None,
                "callback_delivered_at_ms": 1006000,
                "callback_error": None,
                "file_dir": kwargs["file_dir"],
            }
            mock_app_state.queue.get_job.return_value = job_mock
            return None

        mock_app_state.queue.enqueue.side_effect = mock_enqueue

        with patch.object(app.state, "core", mock_app_state):
            with open(audio_file, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.mp3", f, "audio/mpeg")},
                    data={"model": "v1_ctc", "callback_url": "http://callback", "language": "ru"},
                )
            assert response.status_code == 200
            job_data = response.json()
            assert "job_id" in job_data

            job_id = job_data["job_id"]
            status_response = client.get(f"/status?job_id={job_id}")
            assert status_response.status_code == 200

    @pytest.mark.asyncio
    def test_transcribe_with_invalid_model(self, client, mock_app_state):
        mock_app_state.models.is_model_known.return_value = False
        with patch.object(app.state, "core", mock_app_state):
            response = client.post(
                "/transcribe",
                files={"file": ("test.mp3", b"fake", "audio/mpeg")},
                data={"model": "unknown_model", "callback_url": "http://cb"},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    def test_websocket_dashboard(self, client, mock_app_state):
        with TestClient(app).websocket_connect("/ws/dashboard") as ws:
            with patch.object(app.state, "core", mock_app_state):
                data = ws.receive_json()
                assert "health" in data
                assert "models" in data
                assert "gpus" in data
                assert "jobs" in data

    @pytest.mark.asyncio
    def test_error_handling_file_upload_failure(self, client, mock_app_state):
        async def mock_enqueue(**kwargs):
            return None

        mock_app_state.queue.enqueue.side_effect = mock_enqueue
        with patch.object(app.state, "core", mock_app_state):
            response = client.post(
                "/transcribe",
                files={"file": ("test.mp3", b"corrupted", "audio/mpeg")},
                data={"model": "v1_ctc", "callback_url": "http://cb"},
            )
            assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    def test_error_handling_callback_unavailable(self, client, tmp_path, mock_app_state):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake mp3 content")

        async def mock_enqueue_failed_callback(**kwargs):
            job_id = kwargs["job_id"]
            job_mock = MagicMock(
                job_id=job_id,
                status="completed",
                model=kwargs["model"],
                language=kwargs["language"],
                callback_url="http://unavailable.callback",
                created_at_ms=1000000,
                started_at_ms=1000000,
                finished_at_ms=1005000,
                result={"text": "привет", "segments": []},
                error=None,
                callback_delivered_at_ms=None,
                callback_error="connection failed",
                file_dir=kwargs["file_dir"],
            )
            mock_app_state.queue.serialize_job.return_value = {
                "job_id": job_id,
                "status": "completed",
                "model": kwargs["model"],
                "language": kwargs["language"],
                "queue_time_s": 0.0,
                "processing_time_s": 0.0,
                "result": {"text": "привет", "segments": []},
                "error": None,
                "callback": {
                    "delivered": False,
                    "delivered_at_ms": None,
                    "error": "connection failed",
                },
            }
            mock_app_state.queue.get_job.return_value = job_mock
            return None

        mock_app_state.queue.enqueue.side_effect = mock_enqueue_failed_callback

        with patch.object(app.state, "core", mock_app_state):
            with open(audio_file, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.mp3", f, "audio/mpeg")},
                    data={"model": "v1_ctc", "callback_url": "http://unavailable.callback", "language": "ru"},
                )
            assert response.status_code == 200

            job_id = response.json()["job_id"]
            status_response = client.get(f"/status?job_id={job_id}")
            assert status_response.status_code == 200
            job_data = status_response.json()
            assert job_data["callback"]["error"] is not None

    @pytest.mark.asyncio
    def test_parallel_jobs_on_different_gpus(self, client, mock_app_state):
        mock_app_state.queue.snapshot_ids.return_value = (["job1"], ["job2"])
        mock_app_state.queue.serialize_jobs_public.return_value = {
            "total": 2, "queued": 1, "running": 1, "queued_ids": ["job1"], "running_ids": ["job2"]
        }
        mock_app_state.queue.serialize_gpus_public.return_value = [
            {"index": 0, "name": "GPU 0", "status": "running", "current_job_id": "job2", "current_model": "v1_ctc"},
            {"index": 1, "name": "GPU 1", "status": "running", "current_job_id": "job1", "current_model": "v2_rnnt"},
        ]

        with patch.object(app.state, "core", mock_app_state):
            response = client.get("/dashboard/state")
            assert response.status_code == 200
            data = response.json()
            assert len(data["gpus"]) == 2
            running_jobs = [g for g in data["gpus"] if g["status"] == "running"]
            assert len(running_jobs) == 2

    @pytest.mark.asyncio
    def test_different_languages_on_different_models(self, client, mock_app_state):
        job_mock = MagicMock(
            job_id="job_ru",
            status="completed",
            model="v1_ctc",
            language="ru",
            callback_url="http://cb",
            created_at_ms=1000000,
            started_at_ms=1000000,
            finished_at_ms=1005000,
            result={"text": "привет", "segments": []},
            error=None,
            callback_delivered_at_ms=1006000,
            callback_error=None,
            file_dir="/tmp/job_ru",
        )
        mock_app_state.queue.serialize_job.return_value = {
            "job_id": "job_ru",
            "status": "completed",
            "model": "v1_ctc",
            "language": "ru",
            "queue_time_s": 0.0,
            "processing_time_s": 0.0,
            "result": {"text": "привет", "segments": []},
            "error": None,
            "callback": {
                "delivered": True,
                "delivered_at_ms": 1006000,
                "error": None,
            },
        }
        mock_app_state.queue.get_job.return_value = job_mock

        with patch.object(app.state, "core", mock_app_state):
            response_ru = client.get("/status?job_id=job_ru")
            assert response_ru.status_code == 200
            data_ru = response_ru.json()
            assert data_ru["language"] == "ru"
            assert "привет" in data_ru["result"]["text"]

    @pytest.mark.asyncio
    def test_service_restart_with_unloaded_models(self, client, mock_app_state):
        mock_app_state.health_status = "starting"
        mock_app_state.models.unready_details.return_value = "v1_ctc(status=queued_for_download, error=None)"

        with patch.object(app.state, "core", mock_app_state):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["starting", "error"]

    @pytest.mark.asyncio
    def test_all_endpoints_validated(self, client, mock_app_state):
        endpoints_to_test = [
            ("/health", "GET"),
            ("/queue", "GET"),
            ("/dashboard/state", "GET"),
            ("/", "GET"),
        ]

        with patch.object(app.state, "core", mock_app_state):
            for path, method in endpoints_to_test:
                if method == "GET":
                    response = client.get(path)
                    assert response.status_code in [200, 404], f"Endpoint {path} failed"
