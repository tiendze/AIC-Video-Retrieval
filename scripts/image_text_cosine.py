"""CLI: compute cosine similarity between one image and one text query.

The configured multimodal encoder returns L2-normalized vectors, so the dot
product of the image vector and text vector is the cosine similarity.

Prerequisites:
  models/beit3/beit3_base_itc_patch16_224.pth + beit3.spm (scripts/setup_beit3.sh)

Usage:
  python -m scripts.image_text_cosine --image data/keyframes/L21_V001/000001.jpg --text "a red car"
  python -m scripts.image_text_cosine --image 000001.jpg \\
      --image-root data/keyframes --text "a red car"
  python -m scripts.image_text_cosine --image frame.jpg --text "nguoi di xe may" --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from app.encoders.base import MultimodalEncoder
from app.encoders.factory import build_encoder

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ENCODER_CONFIG = _REPO_ROOT / "configs" / "model" / "beit3.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--image", required=True, help="Image path, or filename used with --image-root"
    )
    parser.add_argument("--text", required=True, help="Text query to compare against the image")
    parser.add_argument(
        "--image-root",
        help="Directory searched recursively when --image is only a filename or relative path",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_ENCODER_CONFIG),
        help="Encoder YAML config path. Defaults to configs/model/beit3.yaml",
    )
    parser.add_argument(
        "--device", help="Override encoder device from the config, e.g. cuda or cpu"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_existing_path(path: Path) -> Path:
    return path if path.is_absolute() else _REPO_ROOT / path


def _resolve_image_path(image: str, image_root: str | None = None) -> Path:
    candidate = _resolve_existing_path(Path(image))
    if candidate.is_file():
        return candidate

    if image_root is None:
        raise FileNotFoundError(
            f"image not found: {candidate}. Pass --image-root to search by filename."
        )

    root = _resolve_existing_path(Path(image_root))
    if not root.is_dir():
        raise NotADirectoryError(f"image root not found: {root}")

    rooted_candidate = root / image
    if rooted_candidate.is_file():
        return rooted_candidate

    matches = sorted(path for path in root.rglob(Path(image).name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"image {image!r} not found under {root}")
    if len(matches) > 1:
        preview = ", ".join(str(path.relative_to(root)) for path in matches[:5])
        suffix = "" if len(matches) <= 5 else f", ... ({len(matches)} total)"
        raise ValueError(
            f"image name {image!r} matched multiple files under {root}: {preview}{suffix}"
        )
    return matches[0]


def _cosine_similarity(image_vector: np.ndarray, text_vector: np.ndarray) -> float:
    image_row = np.asarray(image_vector, dtype=np.float32).reshape(-1)
    text_row = np.asarray(text_vector, dtype=np.float32).reshape(-1)
    if image_row.shape != text_row.shape:
        raise ValueError(
            f"vector dimensions do not match: image={image_row.shape}, text={text_row.shape}"
        )
    return float(image_row @ text_row)


def _encode_pair(
    encoder: MultimodalEncoder, image_path: Path, text: str
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        image_vector = encoder.encode_image([image.convert("RGB")])[0]
    text_vector = encoder.encode_text([text])[0]
    return image_vector, text_vector


def _build_result(
    *,
    image_path: Path,
    text: str,
    encoder: MultimodalEncoder,
    config_path: Path,
    cosine: float,
) -> dict[str, Any]:
    return {
        "image": str(image_path),
        "text": text,
        "cosine_similarity": cosine,
        "model_id": encoder.model_id,
        "embedding_dim": encoder.embedding_dim,
        "config": str(config_path),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    config_path = _resolve_existing_path(Path(args.config))

    try:
        image_path = _resolve_image_path(args.image, args.image_root)
        config = _load_yaml(config_path)
        logger.info(
            "loading encoder %s on %s",
            config.get("model_id", config.get("kind", "unknown")),
            args.device or config.get("device", "config-device"),
        )
        encoder = build_encoder(config_path, device=args.device)
        image_vector, text_vector = _encode_pair(encoder, image_path, args.text)
        cosine = _cosine_similarity(image_vector, text_vector)
    except Exception as exc:  # noqa: BLE001 - CLI should report concise user-facing errors.
        logger.error("FAILED: %s", exc)
        return 1

    result = _build_result(
        image_path=image_path,
        text=args.text,
        encoder=encoder,
        config_path=config_path,
        cosine=cosine,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
