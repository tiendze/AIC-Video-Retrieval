"""Encoder factory -- single dispatch point for building a `MultimodalEncoder`
from a `configs/model/*.yaml` file, selected by its `kind` field.

Scope, chốt 2026-08-13 (.claude/plans/encoder-swap-siglip2-pecore.plan.md
"Phạm vi"): the ONLY caller is `scripts/import_embed_out.py`'s `--encoder`
flag. `app/main.py`, `scripts/run_ingest.py`, and `eval/run.py` deliberately
still construct `BEiT3Encoder` directly and do NOT go through this factory --
that is a known, written-down gap (research §7.1's "gom factory 3 chỗ" is
only half-closed here), traded for not touching the running production path
while candidate encoders are still being evaluated.

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
_SUPPORTED_KINDS = ("beit3", "siglip2")


def load_encoder_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_encoder(config_path: Path, device: str | None = None) -> MultimodalEncoder:
    """Builds the encoder described by `config_path`'s `kind` field.

    `device`, given, overrides the config's own `device` value -- same
    override convention as `scripts/embed_clip_baseline.py`'s `--device`.
    """
    config = load_encoder_config(config_path)
    kind = config.get("kind")

    if kind == "beit3":
        from app.encoders.beit3 import BEiT3Encoder

        return BEiT3Encoder(
            checkpoint_path=str(_REPO_ROOT / config["checkpoint_path"]),
            sentencepiece_model_path=str(_REPO_ROOT / config["sentencepiece_model_path"]),
            device=device or config["device"],
            batch_size=config["batch_size"],
            model_id=config["model_id"],
        )

    if kind == "siglip2":
        from app.encoders.siglip2 import SigLip2Encoder

        return SigLip2Encoder(
            hf_repo_id=config["hf_repo_id"],
            hf_revision=config["hf_revision"],
            device=device or config["device"],
            batch_size=config["batch_size"],
            model_id=config["model_id"],
            preprocess_workers=config["preprocess_workers"],
        )

    raise ValueError(
        f"{config_path}: unknown encoder kind {kind!r} -- supported kinds: {_SUPPORTED_KINDS}"
    )
