import os
from pathlib import Path

# kagglehub reads KAGGLEHUB_CACHE at import time, so this must be set
# before `import kagglehub`. Dataset lands under
# <KAGGLEHUB_CACHE>/datasets/<owner>/<dataset>/versions/<version>/
DATA_DIR = Path(__file__).resolve().parent / "data"
os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)

import kagglehub  # noqa: E402

# Download latest version
path = kagglehub.dataset_download("aresusayhi/ai-challenge-2025")

print("Path to dataset files:", path)