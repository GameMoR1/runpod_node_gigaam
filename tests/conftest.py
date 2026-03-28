from __future__ import annotations

import asyncio
import sys

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.types import JobRecord, ModelState


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db_fetch():
    async def _mock_fetch(sql: str, params: dict | None = None):
        return []
    return _mock_fetch


@pytest.fixture
def mock_db_models():
    return [
        {"id_model": 1, "model_name": "v1_ctc"},
        {"id_model": 2, "model_name": "v2_rnnt"},
        {"id_model": 3, "model_name": "v3_conformer"},
    ]


@pytest.fixture
def mock_enabled_ids():
    return [{"model_id": 1}, {"model_id": 2}]


@pytest.fixture
def sample_model_states():
    return [
        ModelState(
            id_model=1,
            model_name="v1_ctc",
            enabled=True,
            status="downloaded",
            progress=100.0,
            error=None,
        ),
        ModelState(
            id_model=2,
            model_name="v2_rnnt",
            enabled=True,
            status="downloading",
            progress=50.0,
            error=None,
        ),
        ModelState(
            id_model=3,
            model_name="v3_conformer",
            enabled=False,
            status="queued_for_download",
            progress=0.0,
            error=None,
        ),
    ]


@pytest.fixture
def sample_job_record():
    return JobRecord(
        job_id="job_123",
        status="queued",
        model="v1_ctc",
        language="ru",
        callback_url="http://callback.test/webhook",
        created_at_ms=1000000,
        started_at_ms=None,
        finished_at_ms=None,
        result=None,
        error=None,
        callback_delivered_at_ms=None,
        callback_error=None,
        file_dir="/tmp/job_123",
    )


@pytest.fixture
def mock_gigaam_assets():
    with patch("app.model_registry.is_gigaam_model") as mock_is, \
         patch("app.model_registry.preload_tokenizer") as mock_preload, \
         patch("app.model_registry.gigaam_cache_dir") as mock_cache:
        mock_is.return_value = True
        mock_cache.return_value = Path("/tmp/gigaam_cache")
        yield mock_is, mock_preload, mock_cache


@pytest.fixture
def mock_gigaam_load():
    mock_gigaam = MagicMock()
    mock_gigaam.load_model.return_value = MagicMock()
    with patch.dict(sys.modules, {'gigaam': mock_gigaam}):
        yield mock_gigaam


@pytest.fixture
def mock_ffmpeg():
    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_subprocess.return_value = mock_proc
        yield mock_subprocess


@pytest.fixture
def temp_job_dir(tmp_path):
    job_dir = tmp_path / "job_123"
    job_dir.mkdir()
    input_file = job_dir / "input"
    input_file.write_bytes(b"fake audio data")
    return job_dir


@pytest.fixture
def sample_wav_file(tmp_path):
    wav_path = tmp_path / "test.wav"
    wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00")
    return str(wav_path)


@pytest.fixture
def mock_gpu_metrics():
    with patch("app.gpu.gpu_metrics") as mock:
        mock.return_value = (50.0, 1024.0, 8192.0)
        yield mock


@pytest.fixture
def mock_gpu_count():
    with patch("app.gpu.gpu_count") as mock:
        mock.return_value = 2
        yield mock

@pytest.fixture
def mock_queueing_gpu():
    with patch("app.queueing.gpu_count", return_value=2), \
         patch("app.queueing.torch_cuda_available", return_value=True), \
         patch("app.queueing.torch_cuda_device_count", return_value=2):
        yield


@pytest.fixture
def mock_torch():
    with patch("app.gigaam_runner.torch") as mock_torch:
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.cuda.reset_peak_memory_stats = MagicMock()
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024
        mock_torch.cuda.empty_cache = MagicMock()
        yield mock_torch


@pytest.fixture
def mock_httpx():
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_wave():
    with patch("wave.open") as mock_wave:
        mock_wf = MagicMock()
        mock_wf.getnframes.return_value = 16000
        mock_wf.getframerate.return_value = 16000
        mock_wave.return_value.__enter__.return_value = mock_wf
        yield mock_wave


@pytest.fixture
def mock_time():
    with patch("app.utils_time.now_ms") as mock_now:
        mock_now.return_value = 1000000
        yield mock_now


@pytest.fixture
def mock_fetch_hugging_face_token():
    with patch("app.db.fetch_hugging_face_token") as mock:
        yield mock

@pytest.fixture
def model_v1_ctc_queued():
    return {
            "v1_ctc": ModelState(
                id_model=1,
                model_name="v1_ctc",
                enabled=True,
                status="queued_for_download",
                progress=0.0,
                error=None,
            )
        }

@pytest.fixture
def model_v1_ctc_and_v2_rnnt_queued():
    return{
            "v1_ctc": ModelState(
                id_model=1,
                model_name="v1_ctc",
                enabled=True,
                status="queued_for_download",
                progress=0.0,
                error=None,
            ),
            "v2_rnnt": ModelState(
                id_model=2,
                model_name="v2_rnnt",
                enabled=True,
                status="queued_for_download",
                progress=0.0,
                error=None,
            ),
        }

@pytest.fixture
def model_v1_ctc_downloaded():
    return{
            "v1_ctc": ModelState(
                id_model=1,
                model_name="v1_ctc",
                enabled=True,
                status="downloaded",
                progress=100.0,
                error=None,
            )
        }

@pytest.fixture
def model_v1_ctc_and_v2_rnnt_downloaded_f():
    return {
            "v1_ctc": ModelState(
                id_model=1,
                model_name="v1_ctc",
                enabled=True,
                status="downloaded",
                progress=100.0,
                error=None,
            ),
            "v2_rnnt": ModelState(
                id_model=2,
                model_name="v2_rnnt",
                enabled=True,
                status="error",
                progress=0.0,
                error="download failed",
            ),
        }
