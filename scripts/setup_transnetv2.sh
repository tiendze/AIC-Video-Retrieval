#!/usr/bin/env bash
# Sets up TransNetV2 for pipelines/shot_detect.py (P0.7).
#
# TransNetV2 has no pip package and no officially-published PyTorch checkpoint
# (verified 2026-08-01 — checked GitHub releases: none; HuggingFace Hub: only
# unofficial third-party mirrors with no checksum, not used here). Instead this
# script reproduces the *official* conversion path end to end:
#   1. Download the TensorFlow SavedModel checkpoint that is committed directly
#      in the soCzech/TransNetV2 repo (inference/transnetv2-weights/).
#   2. Run the repo's own inference-pytorch/convert_weights.py against it,
#      loading the SavedModel with `tf.saved_model.load` (plain TF2 API, not the
#      old TF1-session API despite the repo's dated `tensorflow==2.1` pin) into
#      a throwaway venv, so a modern TensorFlow is fine and the project's main
#      venv never needs TensorFlow at all.
#
# The model architecture source (pipelines/vendor/transnetv2_upstream/) is
# already vendored into this repo (MIT license, soCzech/TransNetV2, fetched
# 2026-08-01) — this script does not clone the repo, only downloads the binary
# weight files and the upstream convert_weights.py.
#
# Runs on the host in this project's shell (matches how scripts/setup_beit3.sh
# is actually run per CLAUDE.md §10 — encoder/model setup is native, not
# containerized; only the three storage services run in Docker per P0.2).
#
# Usage:
#   ./scripts/setup_transnetv2.sh [install_dir]   # default install_dir: models/transnetv2
set -euo pipefail

INSTALL_DIR="${1:-models/transnetv2}"
# The weight files under inference/transnetv2-weights/ are Git-LFS-tracked —
# raw.githubusercontent.com serves the LFS *pointer* text (~130 bytes), not
# the real binary, and a plain `curl --fail` cannot tell (it's a 200 OK).
# media.githubusercontent.com is GitHub's LFS media endpoint and returns the
# actual object; convert_weights.py itself is a normal (non-LFS) text file,
# still fetched from raw.githubusercontent.com.
REPO_RAW_BASE="https://raw.githubusercontent.com/soCzech/TransNetV2/master"
REPO_LFS_BASE="https://media.githubusercontent.com/media/soCzech/TransNetV2/master"
CONVERT_VENV=".venv-transnetv2-convert"
# Pinned for reproducibility (CLAUDE.md §6, "pin version mọi dependency"); any
# TF2 release with `tf.saved_model.load` should work — see header comment.
TF_CONVERT_VERSION="2.17.0"

# Windows Git Bash does not always expose a bare `python` on PATH even with a
# venv active in the calling PowerShell — probe the usual candidates instead
# of hard-failing on the first one that's missing.
PYTHON_BIN=""
for candidate in python python3 py; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    PYTHON_BIN="${candidate}"
    break
  fi
done
if [ -z "${PYTHON_BIN}" ]; then
  echo "ERROR: no python/python3/py found on PATH." >&2
  exit 1
fi
echo "==> Using Python: $(command -v "${PYTHON_BIN}")"

mkdir -p "${INSTALL_DIR}/tf_weights/variables"

echo "==> Downloading TensorFlow SavedModel checkpoint (LFS objects, upstream repo)"
curl -L --fail -o "${INSTALL_DIR}/tf_weights/saved_model.pb" \
  "${REPO_LFS_BASE}/inference/transnetv2-weights/saved_model.pb"
curl -L --fail -o "${INSTALL_DIR}/tf_weights/variables/variables.data-00000-of-00001" \
  "${REPO_LFS_BASE}/inference/transnetv2-weights/variables/variables.data-00000-of-00001"
curl -L --fail -o "${INSTALL_DIR}/tf_weights/variables/variables.index" \
  "${REPO_LFS_BASE}/inference/transnetv2-weights/variables/variables.index"

for f in "${INSTALL_DIR}/tf_weights/saved_model.pb" \
         "${INSTALL_DIR}/tf_weights/variables/variables.data-00000-of-00001" \
         "${INSTALL_DIR}/tf_weights/variables/variables.index"; do
  # An LFS pointer text file is ~130 bytes; every real file here (even the
  # smallest, variables.index, a lightweight name->shape index rather than
  # tensor data) is confirmed several KB, so 2000 cleanly separates the two
  # without being tripped up by variables.index's legitimately modest size.
  size=$(wc -c < "${f}")
  if [ "${size}" -lt 2000 ]; then
    echo "ERROR: ${f} is only ${size} bytes — looks like an LFS pointer or error page, not real weights." >&2
    exit 1
  fi
done

echo "==> Downloading upstream convert_weights.py (run as-is against the vendored model)"
curl -L --fail -o "${INSTALL_DIR}/convert_weights.py" \
  "${REPO_RAW_BASE}/inference-pytorch/convert_weights.py"

echo "==> Creating throwaway venv for one-time TF -> PyTorch weight conversion"
"${PYTHON_BIN}" -m venv "${CONVERT_VENV}"
# shellcheck disable=SC1091
if [ -f "${CONVERT_VENV}/Scripts/activate" ]; then
  source "${CONVERT_VENV}/Scripts/activate"    # Windows venv layout
else
  source "${CONVERT_VENV}/bin/activate"        # POSIX venv layout
fi
pip install --quiet "tensorflow==${TF_CONVERT_VERSION}" torch numpy

echo "==> Converting (imports pipelines/vendor/transnetv2_upstream/transnetv2_pytorch.py)"
PYTHONPATH="pipelines/vendor/transnetv2_upstream" python "${INSTALL_DIR}/convert_weights.py" \
  --tf_weights "${INSTALL_DIR}/tf_weights"
deactivate
mv ./transnetv2-pytorch-weights.pth "${INSTALL_DIR}/transnetv2-pytorch-weights.pth"

echo "==> Cleaning up throwaway venv and TF checkpoint (kept: the converted .pth)"
rm -rf "${CONVERT_VENV}" "${INSTALL_DIR}/tf_weights" "${INSTALL_DIR}/convert_weights.py"

CHECKSUM=$(sha256sum "${INSTALL_DIR}/transnetv2-pytorch-weights.pth" | cut -d' ' -f1)
echo ""
echo "==> Done. Checkpoint: ${INSTALL_DIR}/transnetv2-pytorch-weights.pth"
echo "    SHA256: ${CHECKSUM}"
echo ""
echo "    This hash is self-generated (converted locally from the official TF"
echo "    SavedModel, not an externally-published checksum). After a passing"
echo "    run of tests/test_shot_detect.py, record it in docs/references.md §3.1"
echo "    so a future re-run of this script can detect drift."
echo ""
echo "    Run tests with:"
echo "    TRANSNETV2_CHECKPOINT_PATH=${INSTALL_DIR}/transnetv2-pytorch-weights.pth \\"
echo "    pytest -m model tests/test_shot_detect_model.py -v"
