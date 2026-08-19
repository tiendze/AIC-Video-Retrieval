"""Build an ABE concept bank from SPLiCE-style concept strings.

Default command for the current KIS setup:

    python -m scripts.build_abe_concept_bank --encoder fg_clip

That reads data/abe/splice_concepts.txt, encodes it with models/fgclip_base,
and writes data/abe/concept_bank_fgclip_b16.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from app.encoders.factory import build_encoder_from_config, load_encoder_config
from app.services.abe import (
    AbeConceptBank,
    AbeConceptLookup,
    build_concept_bank_payload,
    load_concept_strings,
    save_concept_bank,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATHS = {
    "beit3": _REPO_ROOT / "configs" / "model" / "beit3.yaml",
    "fg_clip": _REPO_ROOT / "configs" / "model" / "fg_clip.yaml",
    "siglip2": _REPO_ROOT / "configs" / "model" / "siglip2.yaml",
}
_DEFAULT_CONCEPTS_PATH = _REPO_ROOT / "data" / "abe" / "splice_concepts.txt"
_DEFAULT_OUTPUTS = {
    "beit3": _REPO_ROOT / "data" / "abe" / "concept_bank_beit3.pt",
    "fg_clip": _REPO_ROOT / "data" / "abe" / "concept_bank_fgclip_b16.pt",
    "siglip2": _REPO_ROOT / "data" / "abe" / "concept_bank_siglip2.pt",
}
_SANITY_QUERIES = ("shirt", "packet", "car", "helmet")


def main() -> None:
    args = _parse_args()
    config_path, config = _load_requested_config(args)
    encoder_kind = str(config["kind"])
    _apply_cli_overrides(args, config)

    model_id = str(config["model_id"])
    device = _resolve_device(args.device)
    batch_size = args.batch_size
    output_path = _resolve_path(args.output or _DEFAULT_OUTPUTS[encoder_kind])
    concepts_path = _resolve_path(args.concepts)
    checkpoint = _checkpoint_label(config, encoder_kind)

    concepts, duplicates_removed = load_concept_strings(concepts_path)
    print(f"Encoder:                  {encoder_kind}")
    print(f"Model id:                 {model_id}")
    print(f"Device:                   {device}")
    print(f"Concept source:           {concepts_path}")
    print(f"Output path:              {output_path}")
    print("")

    encoder = build_encoder_from_config(
        config,
        config_path=config_path,
        device=device,
        batch_size=batch_size,
    )

    last_reported = -1

    def progress(done: int, total: int) -> None:
        nonlocal last_reported
        percent = int(done * 100 / total)
        if percent != last_reported and (percent % 5 == 0 or done == total):
            print(f"Encoded {done}/{total} concepts ({percent}%)")
            last_reported = percent

    payload = build_concept_bank_payload(
        concepts,
        encoder,
        checkpoint=checkpoint,
        model_id=model_id,
        batch_size=batch_size,
        dtype=args.dtype,
        encoder_kind=encoder_kind,
        variant=config.get("variant"),
        source_vocab=_display_path(concepts_path),
        progress=progress,
    )
    save_concept_bank(payload, output_path)
    bank = AbeConceptBank.from_payload(
        payload,
        expected_model_id=model_id,
        expected_embedding_dim=encoder.embedding_dim,
        expected_encoder_kind=encoder_kind,
        expected_checkpoint=checkpoint,
    )

    embeddings = np.asarray(payload["embeddings"])
    metadata = payload["metadata"]
    print("")
    print(f"Concepts loaded:          {metadata['num_concepts']}")
    print(f"Duplicates removed:       {duplicates_removed}")
    print(f"Embedding dimension:      {metadata['embedding_dim']}")
    print(f"Output dtype:             {metadata['dtype']}")
    print(f"Approx size:              {_format_bytes(embeddings.nbytes)}")
    print("")
    _print_sanity_neighbors(bank, encoder, top_n=args.sanity_top_n)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        choices=tuple(_CONFIG_PATHS),
        default="fg_clip",
        help="Encoder config to use. Defaults to fg_clip.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Override configs/model/<encoder>.yaml.",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=_DEFAULT_CONCEPTS_PATH,
        help="Concept list to encode. Defaults to data/abe/splice_concepts.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .pt path. FG-CLIP defaults to data/abe/concept_bank_fgclip_b16.pt.",
    )
    parser.add_argument("--model-id", help="Override model_id metadata and collection identity.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--sanity-top-n", type=int, default=5)

    # Backward-compatible BEiT3 overrides. FG-CLIP ignores these.
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sentencepiece-model", type=Path)
    return parser.parse_args()


def _load_requested_config(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve_path(args.config or _CONFIG_PATHS[args.encoder])
    config = load_encoder_config(config_path)
    kind = config.get("kind")
    if kind != args.encoder:
        raise ValueError(f"{config_path}: kind={kind!r} does not match --encoder {args.encoder!r}")
    return config_path, config


def _apply_cli_overrides(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.model_id:
        config["model_id"] = args.model_id
    if args.checkpoint:
        if config.get("kind") != "beit3":
            raise ValueError("--checkpoint is only valid with --encoder beit3")
        config["checkpoint_path"] = _display_path(_resolve_path(args.checkpoint))
    if args.sentencepiece_model:
        if config.get("kind") != "beit3":
            raise ValueError("--sentencepiece-model is only valid with --encoder beit3")
        config["sentencepiece_model_path"] = _display_path(_resolve_path(args.sentencepiece_model))


def _checkpoint_label(config: dict[str, Any], encoder_kind: str) -> str:
    if encoder_kind == "fg_clip":
        return _display_path(_resolve_path(config["local_model_path"]))
    if encoder_kind == "beit3":
        return _display_path(_resolve_path(config["checkpoint_path"]))
    if "hf_revision" in config:
        return f"{config.get('hf_repo_id')}@{config['hf_revision']}"
    return str(config.get("model_id", encoder_kind))


def _resolve_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if _cuda_available() else "cpu"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _print_sanity_neighbors(bank: AbeConceptBank, encoder: Any, top_n: int) -> None:
    lookup = AbeConceptLookup(bank, encoder, top_p=top_n)
    for query in _SANITY_QUERIES:
        neighbors = lookup.nearest_concepts(query)
        formatted = ", ".join(
            f"{neighbor.concept} ({neighbor.similarity:.3f})" for neighbor in neighbors
        )
        print(f"Nearest to {query!r}: {formatted}")


if __name__ == "__main__":
    main()
