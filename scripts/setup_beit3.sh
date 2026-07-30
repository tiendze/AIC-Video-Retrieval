#!/usr/bin/env bash
# Sets up BEiT-3 for app/encoders/beit3.py (P0.5).
#
# ⚠️ Resource-heavy (network download ~850MB) — do NOT run on the bare host.
# Run this INSIDE the GPU container that already has the project's Python env,
# per CLAUDE.md §10 ("server không cho cài thư viện trên host").
#
# Usage:
#   ./scripts/setup_beit3.sh [install_dir]   # default install_dir: models/beit3
set -euo pipefail

INSTALL_DIR="${1:-models/beit3}"
RELEASE_BASE="https://github.com/addf400/files/releases/download/beit3"
# Pinned SHA256 of the known-good checkpoint/tokenizer (verified against a
# passing test run) — catches a corrupted or tampered download that would
# otherwise silently load a partially-wrong model (CLAUDE.md §3).
CHECKPOINT_SHA256="d483a41d92b9aea7ef87d17551413b8c22faae927110666b01ac3aa7a199a25b"
SPM_SHA256="6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4"

mkdir -p "${INSTALL_DIR}"

echo "==> Installing pinned BEiT-3 dependencies (pyproject.toml [beit3] extra)"
echo "    NOTE: torch/torchvision are pinned to CUDA 12.1 wheels below — if the"
echo "    container's CUDA runtime differs, adjust --index-url accordingly"
echo "    (check with: python -c 'import torch; print(torch.version.cuda)' if"
echo "    torch is already present, or nvidia-smi for the driver's max supported"
echo "    CUDA version)."
pip install -e ".[beit3]" --extra-index-url https://download.pytorch.org/whl/cu121

echo "==> Downloading beit3_base_itc_patch16_224.pth checkpoint (~850MB)"
curl -L --fail -o "${INSTALL_DIR}/beit3_base_itc_patch16_224.pth" \
  "${RELEASE_BASE}/beit3_base_itc_patch16_224.pth"
echo "${CHECKPOINT_SHA256}  ${INSTALL_DIR}/beit3_base_itc_patch16_224.pth" | sha256sum -c -

echo "==> Downloading beit3.spm tokenizer"
curl -L --fail -o "${INSTALL_DIR}/beit3.spm" "${RELEASE_BASE}/beit3.spm"
echo "${SPM_SHA256}  ${INSTALL_DIR}/beit3.spm" | sha256sum -c -

echo "==> Done. Run tests with:"
echo "    BEIT3_CHECKPOINT_PATH=${INSTALL_DIR}/beit3_base_itc_patch16_224.pth \\"
echo "    BEIT3_SPM_PATH=${INSTALL_DIR}/beit3.spm \\"
echo "    pytest -m model tests/test_beit3_encoder.py -v"
