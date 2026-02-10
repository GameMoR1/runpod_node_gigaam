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
        import contextlib
        import os
        import tempfile
        import wave

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

            def _extract(out_any: Any) -> tuple[str, list[dict[str, Any]]]:
                if isinstance(out_any, str):
                    return out_any, []
                if isinstance(out_any, dict):
                    t = out_any.get("text")
                    text0 = t if isinstance(t, str) else ""
                    seg = out_any.get("segments")
                    segs0 = [s for s in seg if isinstance(s, dict)] if isinstance(seg, list) else []
                    return text0, segs0
                return str(out_any), []

            def _chunk_transcribe(max_seconds: float = 24.0) -> Any:
                tmp_paths: list[str] = []
                try:
                    with contextlib.closing(wave.open(wav_path, "rb")) as wf:
                        nchannels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        framerate = wf.getframerate()
                        nframes = wf.getnframes()
                        chunk_frames = max(1, int(max_seconds * framerate))

                        merged_segments: list[dict[str, Any]] = []

                        offset_s = 0.0
                        frames_left = nframes
                        idx = 0

                        while frames_left > 0:
                            take = min(chunk_frames, frames_left)
                            frames = wf.readframes(take)
                            frames_left -= take

                            with tempfile.NamedTemporaryFile(
                                mode="wb", suffix=f".chunk{idx}.wav", delete=False, dir=os.path.dirname(wav_path)
                            ) as tf:
                                tmp_path = tf.name
                            tmp_paths.append(tmp_path)

                            with contextlib.closing(wave.open(tmp_path, "wb")) as wout:
                                wout.setnchannels(nchannels)
                                wout.setsampwidth(sampwidth)
                                wout.setframerate(framerate)
                                wout.writeframes(frames)

                            part = model.transcribe(tmp_path)
                            part_text, part_segments = _extract(part)

                            if part_segments:
                                for s in part_segments:
                                    ss = dict(s)
                                    for k in ("start", "end"):
                                        v = ss.get(k)
                                        if isinstance(v, (int, float)):
                                            ss[k] = float(v) + offset_s
                                    merged_segments.append(ss)
                            elif part_text:
                                merged_segments.append(
                                    {
                                        "start": offset_s,
                                        "end": offset_s + float(take) / float(framerate),
                                        "text": part_text,
                                        "chunk_index": idx,
                                    }
                                )

                            offset_s += float(take) / float(framerate)
                            idx += 1

                    return {"text": None, "segments": merged_segments}
                finally:
                    for p in tmp_paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass

            try:
                out = model.transcribe(wav_path)
            except ValueError as e:
                msg = str(e)
                if "transcribe_longform" in msg or "Too long" in msg:
                    fn = getattr(model, "transcribe_longform", None)
                    if fn is not None:
                        try:
                            out = fn(wav_path)
                        except Exception:
                            out = _chunk_transcribe()
                    else:
                        out = _chunk_transcribe()
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

        text, segments = _extract(out)

        text_pp = postprocess_text(text)
        if text_pp:
            token_count = len(text_pp.split())
        else:
            token_count = 0
            for s in segments:
                t = s.get("text") if isinstance(s, dict) else None
                if isinstance(t, str) and t:
                    token_count += len(postprocess_text(t).split())
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
