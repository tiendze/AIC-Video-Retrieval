"""Encoder factory -- single dispatch point for building a `MultimodalEncoder`
from a `configs/model/*.yaml` file, selected by its `kind` field.

The live FastAPI app uses this factory to build the current production
semantic encoder from `configs/model/fg_clip.yaml`. Import/eval scripts can
still build older encoders explicitly for offline workflows, but KIS runtime
semantic search is intentionally driven by the FG-CLIP config.

Every branch imports its encoder class LAZILY, never at module level:
importing this module must not require torch/transformers/timm to be
installed, and importing one candidate's dependencies must never pull in
another's. `app/encoders/beit3.py` needs `timm==0.4.12`; `app/encoders/
siglip2.py` needs `transformers>=4.49,<5.0` -- these cannot both be imported
into the same interpreter (see the plan's "Quyết định chiến lược #1").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.encoders.base import MultimodalEncoder

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUPPORTED_KINDS = ("beit3", "fg_clip", "siglip2")


def load_encoder_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_encoder(
    config_path: Path,
    device: str | None = None,
    batch_size: int | None = None,
) -> MultimodalEncoder:
    """Builds the encoder described by `config_path`'s `kind` field.

    `device`, given, overrides the config's own `device` value -- same
    override convention as `scripts/embed_clip_baseline.py`'s `--device`.
    """
    config = load_encoder_config(config_path)
    return build_encoder_from_config(
        config,
        config_path=config_path,
        device=device,
        batch_size=batch_size,
    )


def build_encoder_from_config(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> MultimodalEncoder:
    """Builds an encoder from an already-loaded config mapping."""
    kind = config.get("kind")

    if kind == "beit3":
        from app.encoders.beit3 import BEiT3Encoder

        return BEiT3Encoder(
            checkpoint_path=str(_REPO_ROOT / config["checkpoint_path"]),
            sentencepiece_model_path=str(_REPO_ROOT / config["sentencepiece_model_path"]),
            device=device or config["device"],
            batch_size=batch_size or config["batch_size"],
            model_id=config["model_id"],
        )

    if kind == "fg_clip":
        from app.encoders.fg_clip import FGClipEncoder

        return FGClipEncoder(
            model_path=str(_resolve_repo_path(config["local_model_path"])),
            device=device or config.get("device", "cuda"),
            batch_size=batch_size or config.get("batch_size", 16),
            model_id=config["model_id"],
            image_size=config.get("image_size", 224),
            max_text_tokens=config.get("max_text_tokens", 77),
            embedding_dim=config.get("embedding_dim"),
        )

    if kind == "siglip2":
        from app.encoders.siglip2 import SigLip2Encoder

        return SigLip2Encoder(
            hf_repo_id=config["hf_repo_id"],
            hf_revision=config["hf_revision"],
            device=device or config["device"],
            batch_size=batch_size or config["batch_size"],
            model_id=config["model_id"],
            preprocess_workers=config["preprocess_workers"],
        )

    source = str(config_path) if config_path is not None else "encoder config"
    raise ValueError(
        f"{source}: unknown encoder kind {kind!r} -- supported kinds: {_SUPPORTED_KINDS}"
    )


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO_ROOT / candidate
