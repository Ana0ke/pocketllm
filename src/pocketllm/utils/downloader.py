"""Model downloader - Downloads models from HuggingFace Hub."""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Cache directory for downloaded models
CACHE_DIR = Path.home() / ".cache" / "pocketllm" / "models"


def download_model(
    model_id: str,
    cache_dir: Path | None = None,
) -> Path:
    """Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen2.5-1.5B-Instruct").
        cache_dir: Local cache directory.

    Returns:
        Path to the downloaded model directory.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model: {model_id}")
    model_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        # Only download model files, skip README etc.
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt"],
    )

    return Path(model_path)
