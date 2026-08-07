"""CLI: compare per-video segment counts between two asr_transcripts.json
exports (research-asr-phowhisper-batched.plan.md T6b, N6).

Meant to run right after a re-transcribe under a new ASR config (e.g.
batched=true), comparing against the JSON saved before that switch --
a free check against faster-whisper#1179 (batched inference silently
dropping segments on some inputs). CPU-only, instant: both exports already
carry num_segments per video (pipelines/export_asr.py's build_asr_export),
so no GPU and no re-transcription is needed here.

Usage:
  python scripts/compare_asr_segment_counts.py \
      --before data/asr_transcripts.large-v3.json \
      --after data/asr_transcripts.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pipelines.compare_asr_segment_counts import compare_segment_counts

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--before",
        type=Path,
        default=_REPO_ROOT / "data" / "asr_transcripts.large-v3.json",
        help="baseline export, saved before the config change being checked",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=_REPO_ROOT / "data" / "asr_transcripts.json",
        help="export produced under the new config",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    before = _load_json(args.before)
    after = _load_json(args.after)
    diffs = compare_segment_counts(before, after)

    dropped = [diff for diff in diffs if diff.dropped]
    for diff in diffs:
        logger.info(
            "%-16s before=%-4d after=%-4d delta=%+d%s",
            diff.video_id,
            diff.before,
            diff.after,
            diff.delta,
            "  <-- DROPPED" if diff.dropped else "",
        )

    if dropped:
        logger.warning(
            "%d/%d video(s) lost segments vs %s -- possible faster-whisper#1179 "
            "(batched dropping segments); consider batched=false for these: %s",
            len(dropped),
            len(diffs),
            args.before,
            [diff.video_id for diff in dropped],
        )
        sys.exit(1)

    logger.info("No video lost segments vs %s (%d video compared)", args.before, len(diffs))


if __name__ == "__main__":
    main()
