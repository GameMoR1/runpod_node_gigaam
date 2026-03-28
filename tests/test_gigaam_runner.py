from __future__ import annotations

import sys
import wave
from unittest.mock import MagicMock, patch

import pytest

from app.gigaam_runner import transcribe_on_gpu


@pytest.mark.unit
class TestGigaamRunner:
    @pytest.fixture
    def mock_short_wav(self, tmp_path):
        wav_path = tmp_path / "short.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)
        return str(wav_path)

    @pytest.fixture
    def mock_long_wav(self, tmp_path):
        wav_path = tmp_path / "long.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 960000)
        return str(wav_path)

    @pytest.fixture
    def mock_gigaam_torch(self):
        mock_gigaam = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.cuda.reset_peak_memory_stats = MagicMock()
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024
        mock_torch.cuda.empty_cache = MagicMock()
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3

        with patch.dict(sys.modules, {'gigaam': mock_gigaam, 'torch': mock_torch}):
            yield mock_gigaam, mock_torch

    @pytest.mark.asyncio
    async def test_short_audio_calls_transcribe(
        self, mock_short_wav: str, mock_gpu_metrics, mock_gigaam_torch
    ):
        mock_gigaam, _ = mock_gigaam_torch
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "привет мир", "segments": []}
        mock_gigaam.load_model.return_value = mock_model

        result = await transcribe_on_gpu(
            gpu_index=0,
            wav_path=mock_short_wav,
            model_name="v1_ctc",
            language="ru",
        )

        mock_gigaam.load_model.assert_called_once()
        mock_model.transcribe.assert_called_once()
        assert "text" in result
        assert result["token_count"] >= 0

    @pytest.mark.asyncio
    async def test_long_audio_calls_transcribe_longform(
        self, mock_long_wav: str, mock_gpu_metrics, mock_gigaam_torch
    ):
        mock_gigaam, mock_torch = mock_gigaam_torch
        mock_model = MagicMock()
        mock_model.transcribe_longform.return_value = [
            {"boundaries": [0.0, 10.0], "transcription": "первая часть"},
            {"boundaries": [10.0, 20.0], "transcription": "вторая часть"},
        ]
        mock_gigaam.load_model.return_value = mock_model

        result = await transcribe_on_gpu(
            gpu_index=0,
            wav_path=mock_long_wav,
            model_name="v2_rnnt",
            language="ru",
        )

        mock_model.transcribe_longform.assert_called_once()
        mock_model.transcribe.assert_not_called()
        assert len(result["segments"]) == 2

    @pytest.mark.asyncio
    async def test_long_audio_fallback_to_chunking(
        self, mock_long_wav: str, mock_gpu_metrics, mock_gigaam_torch
    ):
        mock_gigaam, mock_torch = mock_gigaam_torch
        mock_model = MagicMock()
        del mock_model.transcribe_longform
        mock_model.transcribe.return_value = {"text": "часть", "segments": []}
        mock_gigaam.load_model.return_value = mock_model

        result = await transcribe_on_gpu(
            gpu_index=0,
            wav_path=mock_long_wav,
            model_name="v2_rnnt",
            language="ru",
        )

        assert mock_model.transcribe.call_count >= 1
        assert result["text"] is not None

    @pytest.mark.asyncio
    async def test_gpu_metrics_included(
        self, mock_short_wav: str, mock_gpu_metrics, mock_gigaam_torch
    ):
        mock_gigaam, _ = mock_gigaam_torch
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "тест", "segments": []}
        mock_gigaam.load_model.return_value = mock_model

        result = await transcribe_on_gpu(
            gpu_index=0,
            wav_path=mock_short_wav,
            model_name="v1_ctc",
            language="ru",
        )

        assert "gpu" in result
        gpu = result["gpu"]
        assert "util_avg_percent" in gpu
        assert "vram_used_max_mb" in gpu

    @pytest.mark.asyncio
    async def test_postprocess_applied(
        self, mock_short_wav: str, mock_gpu_metrics, mock_gigaam_torch
    ):
        mock_gigaam, _ = mock_gigaam_torch
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "hi\nab\nпривет\nok",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi"},
                {"start": 1.0, "end": 2.0, "text": "привет"},
            ]
        }
        mock_gigaam.load_model.return_value = mock_model

        result = await transcribe_on_gpu(
            gpu_index=0,
            wav_path=mock_short_wav,
            model_name="v1_ctc",
            language="ru",
        )

        text_lines = result["text"].splitlines()
        assert "привет" in text_lines
        assert "hi" not in text_lines

    @pytest.mark.asyncio
    async def test_postprocess_filters_triplet_repeats(
        self, mock_short_wav: str, mock_gpu_metrics, mock_gigaam_torch
    ):
        mock_gigaam, _ = mock_gigaam_torch
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "привет\nааа\nнормально",
            "segments": [],
        }
        mock_gigaam.load_model.return_value = mock_model

        result = await transcribe_on_gpu(
            gpu_index=0,
            wav_path=mock_short_wav,
            model_name="v1_ctc",
            language="ru",
        )

        assert "ааа" not in result["text"]
        assert "нормально" in result["text"]

    @pytest.mark.asyncio
    async def test_token_count_calculated(
        self, mock_short_wav: str, mock_gpu_metrics, mock_gigaam_torch
    ):
        mock_gigaam, _ = mock_gigaam_torch
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "привет мир",
            "segments": []
        }
        mock_gigaam.load_model.return_value = mock_model

        result = await transcribe_on_gpu(
            gpu_index=0,
            wav_path=mock_short_wav,
            model_name="v1_ctc",
            language="ru",
        )

        assert result["token_count"] == 2

    @pytest.mark.asyncio
    async def test_result_structure(
        self, mock_short_wav: str, mock_gpu_metrics, mock_gigaam_torch
    ):
        mock_gigaam, _ = mock_gigaam_torch
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "тест", "segments": []}
        mock_gigaam.load_model.return_value = mock_model

        result = await transcribe_on_gpu(
            gpu_index=0,
            wav_path=mock_short_wav,
            model_name="v1_ctc",
            language="ru",
        )

        required_fields = ["text", "segments", "token_count", "vram_peak_allocated_mb", "language", "gpu"]
        for field in required_fields:
            assert field in result
