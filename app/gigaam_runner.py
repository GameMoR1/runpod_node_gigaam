from __future__ import annotations

import asyncio
from typing import Any

from app.gpu import gpu_metrics
from app.postprocess import postprocess_text


async def transcribe_on_gpu(
    *,
    gpu_index: int,
    wav_path: str,
    model_name: str,
    language: str,
) -> dict[str, Any]:
    util_samples: list[float] = []
    vram_samples: list[float] = []
    vram_total_mb: float = 0.0

    stop = asyncio.Event()

    async def sampler() -> None:
        nonlocal vram_total_mb
        while not stop.is_set():
            util, used_mb, total_mb = gpu_metrics(gpu_index)
            util_samples.append(util)
            vram_samples.append(used_mb)
            if total_mb:
                vram_total_mb = total_mb
            await asyncio.sleep(0.5)

    def run_blocking_sync() -> dict[str, Any]:
        import torch
        import gigaam

        torch.cuda.set_device(gpu_index)
        torch.cuda.reset_peak_memory_stats(gpu_index)

        model = None
        try:
            try:
                model = gigaam.load_model(model_name, device=f"cuda:{gpu_index}")
            except TypeError:
                try:
                    model = gigaam.load_model(model_name, device="cuda")
                except TypeError:
                    model = gigaam.load_model(model_name)

            try:
                out = model.transcribe(wav_path)
            except ValueError as e:
                msg = str(e)
                if "transcribe_longform" in msg or "Too long" in msg:
                    fn = getattr(model, "transcribe_longform", None)
                    if fn is None:
                        raise RuntimeError(
                            "Audio is too long for gigaam.transcribe(); "
                            "install longform deps and use transcribe_longform"
                        ) from e
                    out = fn(wav_path)
                else:
                    raise
        finally:
            try:
                del model
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        text = ""
        segments: list[dict[str, Any]] = []

        if isinstance(out, str):
            text = out
        elif isinstance(out, dict):
            t = out.get("text")
            if isinstance(t, str):
                text = t
            seg = out.get("segments")
            if isinstance(seg, list):
                segments = [s for s in seg if isinstance(s, dict)]
        else:
            text = str(out)

        text_pp = postprocess_text(text)
        token_count = len(text_pp.split()) if text_pp else 0
        peak_alloc_mb = float(torch.cuda.max_memory_allocated(gpu_index)) / (1024 * 1024)

        return {
            "text": text_pp,
            "segments": segments,
            "token_count": token_count,
            "vram_peak_allocated_mb": peak_alloc_mb,
            "language": language,
        }

    sampler_task = asyncio.create_task(sampler())
    try:
        res = await asyncio.to_thread(run_blocking_sync)
    finally:
        stop.set()
        try:
            await sampler_task
        except Exception:
            pass

    util, used_mb, total_mb = gpu_metrics(gpu_index)
    if total_mb and not vram_total_mb:
        vram_total_mb = total_mb

    util_avg = sum(util_samples) / len(util_samples) if util_samples else util
    util_max = max(util_samples) if util_samples else util
    vram_used_avg = sum(vram_samples) / len(vram_samples) if vram_samples else used_mb
    vram_used_max = max(vram_samples) if vram_samples else used_mb
    vram_used_pct_max = (vram_used_max / vram_total_mb * 100.0) if vram_total_mb else 0.0
    vram_used_pct = (used_mb / vram_total_mb * 100.0) if vram_total_mb else 0.0

    res["gpu"] = {
        "index": gpu_index,
        "util_avg_percent": util_avg,
        "util_max_percent": util_max,
        "vram_total_mb": vram_total_mb,
        "vram_used_avg_mb": vram_used_avg,
        "vram_used_max_mb": vram_used_max,
        "vram_used_percent": vram_used_pct,
        "vram_used_percent_max": vram_used_pct_max,
    }
    return res
