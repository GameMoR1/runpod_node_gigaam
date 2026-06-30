from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

GIGAAM_CDN_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
GIGAAM_HF_REPO = "ai-sage/GigaAM-v3"

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

GIGAAM_MODEL_HASHES = {
    "emo": "7ce76f9535cb254488985057c0d33006",
    "v1_ctc": "f027f199e590a391d015aeede2e66174",
    "v1_rnnt": "02c758999bcdc6afcb2087ef256d47ef",
    "v1_ssl": "dc7f7b231f7f91c4968dc21910e7b396",
    "v2_ctc": "e00f59cb5d39624fb30d1786044795bf",
    "v2_rnnt": "547460139acfebd842323f59ed54ab54",
    "v2_ssl": "cd4cf819c8191a07b9d7edcad111668e",
    "v3_ctc": "73413e7be9c6a5935827bfab5c0dd678",
    "v3_rnnt": "0fd2c9a1ff66abd8d32a3a07f7592815",
    "v3_e2e_ctc": "367074d6498f426d960b25f49531cf68",
    "v3_e2e_rnnt": "2730de7545ac43ad256485a462b0a27a",
    "v3_ssl": "70cbf5ed7303a0ed242ddb257e9dc6a6",
}

_V3_SHORT_NAMES = {"ctc", "rnnt", "ssl", "e2e_ctc", "e2e_rnnt"}
_HF_V3_REVISIONS = {
    "v3_ctc": "ctc",
    "v3_rnnt": "rnnt",
    "v3_ssl": "ssl",
    "v3_e2e_ctc": "e2e_ctc",
    "v3_e2e_rnnt": "e2e_rnnt",
    "ctc": "ctc",
    "rnnt": "rnnt",
    "ssl": "ssl",
    "e2e_ctc": "e2e_ctc",
    "e2e_rnnt": "e2e_rnnt",
}

_gigaam_download_patch_applied = False


def is_gigaam_model(model_name: str) -> bool:
    return model_name in GIGAAM_MODEL_NAMES


def gigaam_cache_dir(path: str) -> Path:
    return Path(os.path.expanduser(path))


def canonical_model_name(model_name: str) -> str:
    if model_name in _V3_SHORT_NAMES:
        return f"v3_{model_name}"
    return model_name


def hf_revision_for_model(model_name: str) -> Optional[str]:
    return _HF_V3_REVISIONS.get(model_name) or _HF_V3_REVISIONS.get(canonical_model_name(model_name))


def _needs_tokenizer(model_name: str) -> bool:
    return model_name == "v1_rnnt" or "e2e" in model_name


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _download_cdn_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as source, dest.open("wb") as output:
        while True:
            chunk = source.read(8192)
            if not chunk:
                break
            output.write(chunk)


def _hf_hub_download(*, filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=GIGAAM_HF_REPO,
        filename=filename,
        revision=revision,
        token=_hf_token(),
    )
    return Path(local_path)


def _to_native_state_dict(raw: object) -> dict:
    import torch

    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise RuntimeError("unexpected HuggingFace weights format")

    if not isinstance(state_dict, dict):
        raise RuntimeError("unexpected HuggingFace state_dict format")

    native: dict = {}
    prefixes = ("gigaam.", "model.")
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        native[new_key] = value
    return native


def _build_ckpt_from_hf(*, model_name: str, ckpt_path: Path, revision: str) -> None:
    import omegaconf
    import torch

    config_path = _hf_hub_download(filename="config.json", revision=revision)
    weights_path = _hf_hub_download(filename="pytorch_model.bin", revision=revision)

    with config_path.open(encoding="utf-8") as f:
        hf_config = json.load(f)

    model_cfg = hf_config.get("cfg", {}).get("model", {}).get("cfg")
    if not isinstance(model_cfg, dict):
        raise RuntimeError(f"invalid HuggingFace config for {model_name}")

    cfg = omegaconf.OmegaConf.create(model_cfg)
    state_dict = _to_native_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(".ckpt.part")
    torch.save({"cfg": cfg, "state_dict": state_dict}, tmp_path)
    tmp_path.replace(ckpt_path)

    actual_hash = _file_md5(ckpt_path)
    try:
        import gigaam

        gigaam._MODEL_HASHES[model_name] = actual_hash
    except Exception:
        logger.debug("failed to update gigaam model hash for %s", model_name, exc_info=True)
    logger.info("built gigaam ckpt from HuggingFace: %s (md5=%s)", model_name, actual_hash)


def _download_tokenizer_from_hf(*, model_name: str, dest: Path, revision: str) -> None:
    src = _hf_hub_download(filename="tokenizer.model", revision=revision)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def _download_asset(file_name: str, dest: Path) -> None:
    if dest.exists():
        return

    cdn_url = f"{GIGAAM_CDN_URL.rstrip('/')}/{file_name}"
    cdn_err: Exception | None = None
    try:
        _download_cdn_file(cdn_url, dest)
        logger.info("downloaded gigaam asset from cdn: %s", file_name)
        return
    except Exception as e:
        cdn_err = e
        logger.warning("cdn download failed for %s: %s", file_name, cdn_err)

    if file_name.endswith(".ckpt"):
        model_name = file_name[: -len(".ckpt")]
        revision = hf_revision_for_model(model_name)
        if revision is None:
            raise RuntimeError(f"no HuggingFace fallback for model {model_name}") from cdn_err
        logger.info("downloading gigaam ckpt from HuggingFace: %s (revision=%s)", model_name, revision)
        _build_ckpt_from_hf(model_name=model_name, ckpt_path=dest, revision=revision)
        logger.info("built gigaam ckpt from HuggingFace: %s", file_name)
        return

    if file_name.endswith("_tokenizer.model"):
        model_name = file_name[: -len("_tokenizer.model")]
        revision = hf_revision_for_model(model_name)
        if revision is None:
            raise RuntimeError(f"no HuggingFace fallback for tokenizer {model_name}") from cdn_err
        logger.info("downloading gigaam tokenizer from HuggingFace: %s", file_name)
        _download_tokenizer_from_hf(model_name=model_name, dest=dest, revision=revision)
        return

    raise RuntimeError(f"no HuggingFace fallback for asset {file_name}") from cdn_err


def ensure_gigaam_assets(*, model_name: str, cache_dir: Path) -> None:
    canonical = canonical_model_name(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _download_asset(f"{canonical}.ckpt", cache_dir / f"{canonical}.ckpt")
    if _needs_tokenizer(canonical):
        _download_asset(f"{canonical}_tokenizer.model", cache_dir / f"{canonical}_tokenizer.model")


def preload_tokenizer(*, model_name: str, cache_dir: Path, url_dir: str | None = None) -> Optional[Path]:
    del url_dir
    canonical = canonical_model_name(model_name)
    if not _needs_tokenizer(canonical):
        return None

    tokenizer_path = cache_dir / f"{canonical}_tokenizer.model"
    if tokenizer_path.exists():
        return tokenizer_path

    try:
        _download_asset(f"{canonical}_tokenizer.model", tokenizer_path)
        return tokenizer_path
    except Exception:
        try:
            tokenizer_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def apply_gigaam_download_patch() -> None:
    global _gigaam_download_patch_applied
    if _gigaam_download_patch_applied:
        return

    import gigaam

    def patched_download_file(file_url: str, file_path: str) -> str:
        dest = Path(file_path)
        if dest.exists():
            return file_path

        file_name = file_url.rstrip("/").rsplit("/", 1)[-1]
        _download_asset(file_name, dest)
        return file_path

    gigaam._download_file = patched_download_file
    gigaam._URL_DIR = GIGAAM_CDN_URL
    _gigaam_download_patch_applied = True
    logger.info("gigaam download patch applied (cdn + HuggingFace fallback)")
