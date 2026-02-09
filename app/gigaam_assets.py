from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional


GIGAAM_MODEL_NAMES = {
    "ctc",
    "rnnt",
    "ssl",
    "emo",
    "v1_ssl",
    "v1_ctc",
    "v1_rnnt",
    "v2_ssl",
    "v2_ctc",
    "v2_rnnt",
    "v3_ssl",
    "v3_ctc",
    "v3_rnnt",
    "v3_e2e_ctc",
    "v3_e2e_rnnt",
}


def is_gigaam_model(model_name: str) -> bool:
    return model_name in GIGAAM_MODEL_NAMES


def gigaam_cache_dir(path: str) -> Path:
    return Path(os.path.expanduser(path))


def _needs_tokenizer(model_name: str) -> bool:
    return model_name == "v1_rnnt" or "e2e" in model_name


def preload_tokenizer(*, model_name: str, cache_dir: Path, url_dir: str) -> Optional[Path]:
    if not _needs_tokenizer(model_name):
        return None

    tokenizer_filename = f"{model_name}_tokenizer.model"
    tokenizer_path = cache_dir / tokenizer_filename
    if tokenizer_path.exists():
        return tokenizer_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_url = f"{url_dir.rstrip('/')}/{tokenizer_filename}"
    try:
        urllib.request.urlretrieve(tokenizer_url, str(tokenizer_path))
        return tokenizer_path
    except Exception:
        try:
            tokenizer_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

