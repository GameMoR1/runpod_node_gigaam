from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.gpu import (
    gpu_count,
    gpu_metrics,
    gpu_name,
    torch_cuda_available,
    torch_cuda_device_count,
)

@pytest.mark.unit
class TestGpu:
    def test_gpu_count_with_nvml(self):
        with patch("app.gpu._NVML_OK", True):
            with patch("pynvml.nvmlDeviceGetCount", return_value=4):
                assert gpu_count() == 4

    def test_gpu_count_fallback_to_torch(self):
        with patch("app.gpu._NVML_OK", False):
            with patch("app.gpu._HAS_TORCH", True):
                with patch("torch.cuda.device_count", return_value=2):
                    assert gpu_count() == 2

    def test_gpu_count_returns_zero_when_no_gpu(self):
        with patch("app.gpu._NVML_OK", False):
            with patch("app.gpu._HAS_TORCH", False):
                assert gpu_count() == 0

    def test_torch_cuda_available(self):
        with patch("app.gpu._HAS_TORCH", True):
            with patch("torch.cuda.is_available", return_value=True):
                assert torch_cuda_available() is True

    def test_torch_cuda_device_count(self):
        with patch("app.gpu._HAS_TORCH", True):
            with patch("torch.cuda.device_count", return_value=8):
                assert torch_cuda_device_count() == 8

    def test_gpu_name_with_torch(self):
        with patch("app.gpu._HAS_TORCH", True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                assert gpu_name(0) == "NVIDIA RTX 4090"

    def test_gpu_name_fallback(self):
        with patch("app.gpu._HAS_TORCH", False):                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            assert gpu_name(1) == "GPU 1"

    def test_gpu_metrics_with_nvml(self):
        with patch("app.gpu._NVML_OK", True):
            mock_handle = MagicMock()
            mock_util = MagicMock()
            mock_util.gpu = 75.5
            mock_mem = MagicMock()
            mock_mem.used = 4 * 1024**3
            mock_mem.total = 8 * 1024**3
            mock_handle.return_value = mock_handle
            with patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
                with patch("pynvml.nvmlDeviceGetUtilizationRates", return_value=mock_util):
                    with patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_mem):
                        util, used, total = gpu_metrics(0)
                        assert util == 75.5
                        assert used == 4096.0
                        assert total == 8192.0

    def test_gpu_metrics_fallback_to_torch(self):
        with patch("app.gpu._NVML_OK", False):
            with patch("app.gpu._HAS_TORCH", True):
                mock_props = MagicMock()
                mock_props.total_memory = 8 * 1024**3
                with patch("torch.cuda.set_device"):
                    with patch("torch.cuda.get_device_properties", return_value=mock_props):
                        with patch("torch.cuda.memory_allocated", return_value=2 * 1024**3):
                            util, used, total = gpu_metrics(0)
                            assert total == 8192.0
                            assert used == 2048.0

    def test_gpu_metrics_returns_zeros_on_error(self):
        with patch("app.gpu._NVML_OK", False):
            with patch("app.gpu._HAS_TORCH", False):
                util, used, total = gpu_metrics(0)
                assert (0.0, 0.0, 0.0) == (util, used, total)
