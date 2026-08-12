"""CLI: verify a Kaggle-produced embed_out/ artifact is safe to import.

Gate required by kaggle-offline-embed.plan.md §3/§6 before any import code
reads data/embed_out/vectors/ -- checks the local BEiT-3 checkpoint/tokenizer
match what produced the artifact, then re-encodes the manifest's probe texts
locally and requires cosine >= 0.9999 against Kaggle's probe vectors. Either
check failing exits non-zero; nothing gets silently accepted.

Prerequisites:
  models/beit3/beit3_base_itc_patch16_224.pth + beit3.spm (scripts/setup_beit3.sh)
  data/embed_out/ downloaded and unzipped from Kaggle

WARNING: GPU-heavy (loads the BEiT-3 checkpoint) -- per CLAUDE.md §10,
prepared here for you to run yourself.

Usage:
  python scripts/verify_embed_parity.py
  python scripts/verify_embed_parity.py --embed-out-dir data/embed_out --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from pipelines.verify_embed_parity import (
    EmbedParityError,
    ParityReport,
    load_manifest,
    load_probe,
    run_parity_probe,
    verify_checksums,
    verify_manifest_fields,
)

# Windows consoles often default to cp1252, which cannot encode the
# Vietnamese probe text this script logs -- same fix as run_ingest.py.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BEIT3_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "beit3.yaml"
_COSINE_THRESHOLD = 0.9999  # kaggle-offline-embed.plan.md §3 "Parity probe"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--embed-out-dir", default=str(_REPO_ROOT / "data" / "embed_out"), help="Path to embed_out/"
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_report(report: ParityReport) -> None:
    for result in report.probe_results:
        marker = "OK" if result.cosine >= report.threshold else "FAIL"
        logger.info("[%s] cosine=%.6f  %s", marker, result.cosine, result.text)
    logger.info(
        "min cosine=%.6f  threshold=%.6f  %s",
        report.min_cosine,
        report.threshold,
        "PASSED" if report.passed else "FAILED",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    embed_out_dir = Path(args.embed_out_dir)
    beit3_config = _load_yaml(_BEIT3_CONFIG_PATH)

    manifest = load_manifest(embed_out_dir / "manifest.json")

    try:
        verify_checksums(
            manifest,
            checkpoint_path=_REPO_ROOT / beit3_config["checkpoint_path"],
            spm_path=_REPO_ROOT / beit3_config["sentencepiece_model_path"],
        )
    except EmbedParityError as exc:
        logger.error("BLOCKED: %s", exc)
        sys.exit(1)
    logger.info("checkpoint_sha256 / spm_sha256: OK")

    from app.encoders.beit3 import BEiT3Encoder

    encoder = BEiT3Encoder(
        checkpoint_path=str(_REPO_ROOT / beit3_config["checkpoint_path"]),
        sentencepiece_model_path=str(_REPO_ROOT / beit3_config["sentencepiece_model_path"]),
        device=args.device,
        batch_size=beit3_config["batch_size"],
        model_id=beit3_config["model_id"],
    )

    try:
        verify_manifest_fields(
            manifest, model_id=encoder.model_id, embedding_dim=encoder.embedding_dim
        )
    except EmbedParityError as exc:
        logger.error("BLOCKED: %s", exc)
        sys.exit(1)
    logger.info("model_id / embedding_dim: OK")

    texts, remote_vectors = load_probe(embed_out_dir / "probe")
    probe_results = run_parity_probe(texts, remote_vectors, encoder.encode_text)
    report = ParityReport(probe_results=probe_results, threshold=_COSINE_THRESHOLD)
    _print_report(report)

    if not report.passed:
        logger.error("BLOCKED: parity probe failed -- do not import data/embed_out/vectors/")
        sys.exit(1)

    logger.info("Parity probe passed -- embed_out/ is safe to import.")


if __name__ == "__main__":
    main()
