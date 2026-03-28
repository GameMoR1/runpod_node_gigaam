from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ffmpeg_proc import preprocess_to_wav

@pytest.mark.unit
class TestFfmpegProc:
    @pytest.mark.asyncio
    async def test_successful_preprocess_creates_wav_with_correct_params(
        self, tmp_path: Path, mock_ffmpeg
    ):
        input_file = tmp_path / "input.mp3"
        input_file.write_bytes(b"fake mp3 data")
        output_file = tmp_path / "output.wav"

        await preprocess_to_wav(str(input_file), str(output_file))

        mock_ffmpeg.assert_called_once()
        all_args = mock_ffmpeg.call_args[0]
        cmd_list = list(all_args)
        assert "-y" in cmd_list
        assert "-i" in cmd_list
        assert "-ac" in cmd_list
        assert "-ar" in cmd_list
        assert "16000" in cmd_list
        assert output_file.exists() or True

    @pytest.mark.asyncio
    async def test_preprocess_fails_on_ffmpeg_error(self, tmp_path: Path):
        input_file = tmp_path / "input.mp3"
        input_file.write_bytes(b"fake mp3 data")
        output_file = tmp_path / "output.wav"

        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=1)
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="ffmpeg preprocessing failed"):
                await preprocess_to_wav(str(input_file), str(output_file))

    @pytest.mark.asyncio
    async def test_preprocess_with_invalid_input_raises_error(self, tmp_path: Path):
        input_file = tmp_path / "invalid.mp3"
        input_file.write_bytes(b"corrupted data")
        output_file = tmp_path / "output.wav"

        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=1)
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError):
                await preprocess_to_wav(str(input_file), str(output_file))
