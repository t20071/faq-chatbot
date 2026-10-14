"""Fetch the ONNX model files from Hugging Face if they're missing.

The files are committed in chatbot/models/ and bundled with the Vercel function,
so normally nothing is downloaded. This is the fallback for a fresh checkout without
them: download once (into /tmp if the filesystem is read-only), pinned to a revision.
"""

import logging
import os
import tempfile
import urllib.request
from pathlib import Path

from faqbot.config import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    MODEL_DIR,
    ONNX_MODEL_FILE,
    ONNX_REPO_FILE,
)

log = logging.getLogger(__name__)

# local file name -> path inside the Hugging Face repo
FILES = {"tokenizer.json": "tokenizer.json", ONNX_MODEL_FILE: ONNX_REPO_FILE}


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return os.access(directory, os.W_OK)
    except OSError:
        return False


def ensure_model_files(model_dir: Path = MODEL_DIR) -> Path:
    """Return a directory containing the model files, downloading any that are missing."""
    if all((model_dir / name).exists() for name in FILES):
        return model_dir
    if not _writable(model_dir):  # read-only deployment filesystem (Vercel)
        model_dir = Path(tempfile.gettempdir()) / "models" / model_dir.name
        model_dir.mkdir(parents=True, exist_ok=True)

    for name, repo_path in FILES.items():
        target = model_dir / name
        if target.exists():
            continue
        url = (
            f"https://huggingface.co/{EMBEDDING_MODEL}/resolve/"
            f"{EMBEDDING_MODEL_REVISION}/{repo_path}"
        )
        log.info("Downloading %s", url)
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)  # atomic: never leave a half-written model behind
    return model_dir
