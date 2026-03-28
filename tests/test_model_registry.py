from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from app.config import settings
from app.model_registry import ModelRegistry

@pytest.mark.unit
class TestModelRegistry:
    @pytest.fixture
    def registry(self):
        return ModelRegistry()

    @pytest.mark.asyncio
    async def test_load_from_db_and_prepare_fetches_gigaam_models(
        self, registry: ModelRegistry, mock_db_models: list[dict], mock_enabled_ids: list[dict], mock_gigaam_assets, mock_gigaam_load
    ):
        with patch("app.model_registry.fetch_all", side_effect=[mock_db_models, mock_enabled_ids]):
            await registry.load_from_db_and_prepare()

        assert len(registry._models) == 2
        assert "v1_ctc" in registry._models
        assert "v2_rnnt" in registry._models
        assert "v3_conformer" not in registry._models

    @pytest.mark.asyncio
    async def test_load_from_db_filters_only_enabled_models(
        self, registry: ModelRegistry, mock_db_models: list[dict], mock_gigaam_assets
    ):
        enabled_ids = [{"model_id": 1}]
        with patch("app.model_registry.fetch_all", side_effect=[mock_db_models, enabled_ids]), \
             patch.object(registry, "_download_model", return_value=None):
            await registry.load_from_db_and_prepare()

        assert len(registry._models) == 1
        assert registry._models["v1_ctc"].enabled is True
        assert registry._models["v1_ctc"].status == "queued_for_download"

    @pytest.mark.asyncio
    async def test_download_model_success_status_transitions(
        self, registry: ModelRegistry, mock_gigaam_assets, mock_gigaam_load, model_v1_ctc_queued):
        registry._models = model_v1_ctc_queued

        await registry._download_model("v1_ctc")

        model = registry._models["v1_ctc"]
        assert model.status == "downloaded"
        assert model.progress == 100.0
        assert model.error is None

    @pytest.mark.asyncio
    async def test_download_model_failure_sets_error_status(
        self, registry: ModelRegistry, mock_gigaam_assets, model_v1_ctc_queued):
        registry._models = model_v1_ctc_queued

        mock_gigaam = MagicMock()
        mock_gigaam.load_model.side_effect = RuntimeError("download failed")
        with patch.dict(sys.modules, {'gigaam': mock_gigaam}):
            await registry._download_model("v1_ctc")

        model = registry._models["v1_ctc"]
        assert model.status == "error"
        assert model.error is not None

    @pytest.mark.asyncio
    async def test_download_model_retries_on_failure(
        self, registry: ModelRegistry, mock_gigaam_assets, model_v1_ctc_queued):
        registry._models = model_v1_ctc_queued

        mock_gigaam = MagicMock()
        mock_gigaam.load_model.side_effect = [RuntimeError("fail 1"), RuntimeError("fail 2"), MagicMock()]
        with patch.dict(sys.modules, {'gigaam': mock_gigaam}):
            with patch.object(settings, "MODEL_DOWNLOAD_ATTEMPTS", "3"):
                await registry._download_model("v1_ctc")

        model = registry._models["v1_ctc"]
        assert model.status == "downloaded"
        assert mock_gigaam.load_model.call_count == 3

    @pytest.mark.asyncio
    async def test_parallel_download_two_models(
        self, registry: ModelRegistry, mock_gigaam_assets, mock_gigaam_load, model_v1_ctc_and_v2_rnnt_queued):
        registry._models = model_v1_ctc_and_v2_rnnt_queued

        with patch.dict(sys.modules, {'gigaam': mock_gigaam_load}):
            await asyncio.gather(
                registry._download_model("v1_ctc"),
                registry._download_model("v2_rnnt"),
            )

        assert registry._models["v1_ctc"].status == "downloaded"
        assert registry._models["v2_rnnt"].status == "downloaded"

    @pytest.mark.asyncio
    async def test_already_downloaded_model_skipped(
        self, registry: ModelRegistry, tmp_path: Path, model_v1_ctc_downloaded):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        model_dir = cache_dir / "v1_ctc"
        model_dir.mkdir()
        (model_dir / "model.pt").write_text("fake model")

        registry._models = model_v1_ctc_downloaded

        with patch.object(settings, "MODEL_CACHE_DIR", str(cache_dir)):
            mock_gigaam = MagicMock()
            mock_gigaam.load_model.return_value = MagicMock()
            with patch.dict(sys.modules, {'gigaam': mock_gigaam}):
                await registry._download_model("v1_ctc")

        assert registry._models["v1_ctc"].status == "downloaded"

    def test_is_model_known_returns_true_for_downloaded(
        self, registry: ModelRegistry, model_v1_ctc_downloaded):
        registry._models = model_v1_ctc_downloaded

        assert registry.is_model_known("v1_ctc") is True
        assert registry.is_model_known("unknown") is False

    def test_unready_models_returns_non_downloaded(
        self, registry: ModelRegistry, model_v1_ctc_and_v2_rnnt_downloaded_f):
        registry._models = model_v1_ctc_and_v2_rnnt_downloaded_f

        unready = registry.unready_models()
        assert len(unready) == 1
        assert unready[0].model_name == "v2_rnnt"


class TestAppState:
    @pytest.mark.asyncio
    async def test_fetch_hugging_face_token_sets_env_vars(self, mock_fetch_hugging_face_token):
        from app.state import AppState
        mock_fetch_hugging_face_token.return_value = "hf_token_123"
        state = AppState()
        state.models = MagicMock()
        state.queue = MagicMock()
        state.models.load_from_db_and_prepare = AsyncMock()
        state.queue.start_workers = AsyncMock()
        state.models.unready_details.return_value = ""
        with patch('app.state._check_ffmpeg'), \
             patch('app.state._check_gigaam_module'):
            await state._initialize()
        assert os.environ.get("HF_TOKEN") == "hf_token_123"
        assert os.environ.get("HUGGINGFACEHUB_API_TOKEN") == "hf_token_123"
