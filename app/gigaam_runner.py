from __future__ import annotations

import asyncio
from typing import Any

from app.gpu import gpu_metrics
from app.postprocess import postprocess_text
from app.vad_chunking import AudioChunk


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
        import os
        import subprocess
        import tempfile
        import wave

        from app.config import settings

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

            def _duration_s() -> float:
                try:
                    with wave.open(wav_path, "rb") as wf:
                        fr = wf.getframerate()
                        nframes = wf.getnframes()
                        if fr <= 0:
                            return 0.0
                        return float(nframes) / float(fr)
                except Exception:
                    return 0.0

            def _assert_nonempty_wav(path: str) -> None:
                with wave.open(path, "rb") as wf:
                    if wf.getnframes() <= 0 or wf.getframerate() <= 0:
                        raise RuntimeError("empty wav audio")

            def _extract_text(out_any: Any) -> str:
                if isinstance(out_any, str):
                    return out_any
                if isinstance(out_any, dict):
                    t = out_any.get("text")
                    return t if isinstance(t, str) else ""
                return str(out_any)

            def _segments_from_longform(utterances: Any) -> list[dict[str, Any]]:
                segs: list[dict[str, Any]] = []
                if not isinstance(utterances, list):
                    return segs
                for utt in utterances:
                    if not isinstance(utt, dict):
                        continue
                    b = utt.get("boundaries")
                    if not (isinstance(b, (list, tuple)) and len(b) >= 2):
                        continue
                    start, end = b[0], b[1]
                    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                        continue
                    txt = utt.get("transcription")
                    segs.append(
                        {
                            "start": float(start),
                            "end": float(end),
                            "text": str(txt) if isinstance(txt, str) else "",
                        }
                    )
                return segs

            def _chunk_transcribe(*, chunk_duration_s: float = 20.0) -> tuple[str, list[dict[str, Any]]]:
                total = _duration_s()
                if total <= 0.01:
                    raise RuntimeError("empty audio after preprocessing")
                start = 0.0
                segs: list[dict[str, Any]] = []

                while start + 0.01 < total:
                    end = min(start + chunk_duration_s, total)
                    if end - start <= 0.01:
                        break
                    fd, chunk_path = tempfile.mkstemp(suffix=".wav", dir=os.path.dirname(wav_path) or None)
                    os.close(fd)
                    try:
                        cmd_re = [
                            settings.FFMPEG_PATH,
                            "-y",
                            "-loglevel",
                            "error",
                            "-ss",
                            str(start),
                            "-t",
                            str(end - start),
                            "-i",
                            wav_path,
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            "-c:a",
                            "pcm_s16le",
                            chunk_path,
                        ]
                        subprocess.run(cmd_re, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        _assert_nonempty_wav(chunk_path)

                        part = model.transcribe(chunk_path)
                        part_text = _extract_text(part)
                        segs.append({"start": float(start), "end": float(end), "text": part_text})
                    finally:
                        try:
                            os.remove(chunk_path)
                        except Exception:
                            pass
                    start = end

                full_text = " ".join(s.get("text") or "" for s in segs).strip()
                return full_text, segs

            duration_s = _duration_s()
            if duration_s <= 0.01:
                raise RuntimeError("empty audio after preprocessing")
            if duration_s and duration_s <= 25.0:
                out_short = model.transcribe(wav_path)
                text0 = _extract_text(out_short)
                segs0: list[dict[str, Any]] = []
                if isinstance(out_short, dict) and isinstance(out_short.get("segments"), list):
                    segs0 = [s for s in out_short.get("segments") if isinstance(s, dict)]
                if not segs0:
                    segs0 = [{"start": 0.0, "end": float(duration_s), "text": text0}]
                out = {"text": text0, "segments": segs0}
            else:
                fn = getattr(model, "transcribe_longform", None)
                if fn is None:
                    text0, segs0 = _chunk_transcribe()
                    out = {"text": text0, "segments": segs0}
                else:
                    try:
                        utterances = fn(wav_path)
                        segs0 = _segments_from_longform(utterances)
                        if not segs0:
                            raise RuntimeError("empty longform result")
                        text0 = " ".join(s.get("text") or "" for s in segs0).strip()
                        out = {"text": text0, "segments": segs0}
                    except Exception:
                        text0, segs0 = _chunk_transcribe()
                        out = {"text": text0, "segments": segs0}
        finally:
            try:
                del model
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        text = out.get("text") if isinstance(out, dict) else ""
        segments = out.get("segments") if isinstance(out, dict) and isinstance(out.get("segments"), list) else []
        segments = [s for s in segments if isinstance(s, dict)]

        for s in segments:
            t = s.get("text")
            if isinstance(t, str):
                s["text"] = postprocess_text(t)

        text_pp = postprocess_text(text if isinstance(text, str) else str(text))
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


async def transcribe_chunks_on_gpu(
    *,
    gpu_index: int,
    chunks: list[AudioChunk],
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
        import os
        import subprocess
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

            segs: list[dict[str, Any]] = []
            texts: list[str] = []

            from app.config import settings

            def _wav_duration_s(path: str) -> float:
                try:
                    with wave.open(path, "rb") as wf:
                        fr = wf.getframerate()
                        nframes = wf.getnframes()
                        if fr <= 0:
                            return 0.0
                        return float(nframes) / float(fr)
                except Exception:
                    return 0.0

            def _segments_from_longform(utterances: Any) -> list[dict[str, Any]]:
                out: list[dict[str, Any]] = []
                if not isinstance(utterances, list):
                    return out
                for utt in utterances:
                    if not isinstance(utt, dict):
                        continue
                    b = utt.get("boundaries")
                    if not (isinstance(b, (list, tuple)) and len(b) >= 2):
                        continue
                    start, end = b[0], b[1]
                    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                        continue
                    txt = utt.get("transcription")
                    out.append(
                        {
                            "start": float(start),
                            "end": float(end),
                            "text": str(txt) if isinstance(txt, str) else "",
                        }
                    )
                return out

            def _extract_text(out_any: Any) -> str:
                if isinstance(out_any, str):
                    return out_any
                if isinstance(out_any, dict):
                    t = out_any.get("text")
                    return t if isinstance(t, str) else ""
                return str(out_any)

            def _chunk_file_transcribe(path: str, *, chunk_duration_s: float = 20.0) -> tuple[str, list[dict[str, Any]]]:
                total = _wav_duration_s(path)
                if total <= 0.01:
                    return "", []
                start = 0.0
                parts: list[dict[str, Any]] = []
                while start + 0.01 < total:
                    end = min(start + chunk_duration_s, total)
                    if end - start <= 0.01:
                        break
                    fd, chunk_path = tempfile.mkstemp(suffix=".wav", dir=os.path.dirname(path) or None)
                    os.close(fd)
                    try:
                        cmd = [
                            settings.FFMPEG_PATH,
                            "-y",
                            "-loglevel",
                            "error",
                            "-ss",
                            str(start),
                            "-t",
                            str(end - start),
                            "-i",
                            path,
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            "-c:a",
                            "pcm_s16le",
                            chunk_path,
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        out_any = model.transcribe(chunk_path)
                        t = _extract_text(out_any)
                        parts.append({"start": float(start), "end": float(end), "text": t})
                    finally:
                        try:
                            os.remove(chunk_path)
                        except Exception:
                            pass
                    start = end
                full = " ".join(p.get("text") or "" for p in parts).strip()
                return full, parts

            def _safe_transcribe(path: str) -> tuple[str, list[dict[str, Any]]]:
                """
                GigaAM throws ValueError for too-long audio in transcribe().
                Use transcribe_longform() when available; otherwise fall back to internal splitting.
                """
                dur = _wav_duration_s(path)
                fn = getattr(model, "transcribe_longform", None)
                if dur > 25.0 and fn is not None:
                    utterances = fn(path)
                    segs0 = _segments_from_longform(utterances)
                    if segs0:
                        txt0 = " ".join(s.get("text") or "" for s in segs0).strip()
                        return txt0, segs0

                try:
                    out_any = model.transcribe(path)
                    txt = _extract_text(out_any)
                    segs0 = []
                    if isinstance(out_any, dict) and isinstance(out_any.get("segments"), list):
                        segs0 = [s for s in out_any.get("segments") if isinstance(s, dict)]
                    if segs0:
                        return txt, segs0
                    return txt, []
                except ValueError as e:
                    msg = str(e)
                    if "Too long wav file" in msg and fn is not None:
                        utterances = fn(path)
                        segs0 = _segments_from_longform(utterances)
                        txt0 = " ".join(s.get("text") or "" for s in segs0).strip()
                        return txt0, segs0
                    if "Too long wav file" in msg:
                        return _chunk_file_transcribe(path)
                    raise

            for chunk in chunks:
                t0, seg0 = _safe_transcribe(str(chunk.path))

                t_pp = postprocess_text(t0).strip()
                if t_pp:
                    texts.append(t_pp)

                if seg0:
                    for s in seg0:
                        st = s.get("start")
                        en = s.get("end")
                        if isinstance(st, (int, float)) and isinstance(en, (int, float)):
                            s2 = dict(s)
                            s2["start"] = float(st) + float(chunk.start)
                            s2["end"] = float(en) + float(chunk.start)
                            tt = s2.get("text")
                            if isinstance(tt, str):
                                s2["text"] = postprocess_text(tt)
                            segs.append(s2)
                else:
                    if t_pp:
                        segs.append({"start": float(chunk.start), "end": float(chunk.end), "text": t_pp})

            full_text = " ".join(texts).strip()
        finally:
            try:
                del model
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        peak_alloc_mb = float(torch.cuda.max_memory_allocated(gpu_index)) / (1024 * 1024)
        token_count = len(full_text.split()) if full_text else 0
        return {
            "text": full_text,
            "segments": segs,
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
