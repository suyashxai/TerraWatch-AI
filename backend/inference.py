"""
inference.py
------------
Loads Prithvi-EO-2.0-300M-BurnScars and runs semantic segmentation.

Architecture (from burn_scars_config.yaml):
  - backbone  : prithvi_eo_v2_300
  - necks     : SelectIndices([5,11,17,23]) -> ReshapeTokensToImage -> LearnedInterpolateToPyramidal
  - decoder   : UNetDecoder  channels=[512,256,128,64]
  - num_classes: 2  (0=not burned, 1=burn scar)

Loaded via terratorch.models.EncoderDecoderFactory (official path).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Suppress harmless torch.jit deprecation warning from terratorch internals
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logger = logging.getLogger(__name__)

MODELS_DIR       = Path(__file__).resolve().parent.parent / "models"
LOCAL_WEIGHTS    = MODELS_DIR / "Prithvi_EO_V2_300M_BurnScars.pt"
HF_REPO_ID       = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
WEIGHTS_FILENAME = "Prithvi_EO_V2_300M_BurnScars.pt"

logger.debug("MODELS_DIR resolved to: %s  (exists=%s)", MODELS_DIR, MODELS_DIR.exists())
logger.debug("LOCAL_WEIGHTS path: %s  (exists=%s)", LOCAL_WEIGHTS, LOCAL_WEIGHTS.exists())


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_available() -> bool:
    return LOCAL_WEIGHTS.exists()


# ---------------------------------------------------------------------------
# Weight download helper
# ---------------------------------------------------------------------------

def _ensure_weights() -> Path:
    """
    Return a path to the weights file.

    Resolution order:
    1. models/Prithvi_EO_V2_300M_BurnScars.pt  (committed or manually placed)
    2. HF hub cache (already-downloaded, avoids redundant network calls)
    3. Fresh download from Hugging Face Hub
    """
    # 1. Preferred: file committed into the repo / placed in models/
    if LOCAL_WEIGHTS.exists():
        logger.info("Using local weights: %s", LOCAL_WEIGHTS)
        return LOCAL_WEIGHTS

    # 2. Check the HF hub cache without hitting the network
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        cached = try_to_load_from_cache(
            repo_id=HF_REPO_ID,
            filename=WEIGHTS_FILENAME,
        )
        if cached and Path(cached).exists():
            logger.info("Using HF-cached weights: %s", cached)
            return Path(cached)
    except Exception:
        pass  # function may not exist in older huggingface_hub versions

    # 3. Download (first run on a fresh environment)
    logger.info("Weights not found locally — downloading from Hugging Face...")
    from huggingface_hub import hf_hub_download  # type: ignore
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=WEIGHTS_FILENAME,
        local_dir=str(MODELS_DIR),
    )
    return Path(path)


# ---------------------------------------------------------------------------
# Model loading via TerraTorch EncoderDecoderFactory
# ---------------------------------------------------------------------------

def load_model(device: Optional[torch.device] = None):
    """
    Build and load the Prithvi-EO-2.0-300M-BurnScars model using TerraTorch's
    EncoderDecoderFactory, matching burn_scars_config.yaml exactly.

    Raises RuntimeError with a clear message on failure.
    """
    if device is None:
        device = get_device()

    weights_path = _ensure_weights()

    try:
        from terratorch.models import EncoderDecoderFactory  # type: ignore

        factory = EncoderDecoderFactory()
        model = factory.build_model(
            task="segmentation",
            backbone="prithvi_eo_v2_300",
            backbone_pretrained=False,           # we load weights manually below
            backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            necks=[
                {"name": "SelectIndices",              "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            decoder="UNetDecoder",
            decoder_channels=[512, 256, 128, 64],
            num_classes=2,
        )
    except Exception as exc:
        raise RuntimeError(
            f"TerraTorch model build failed: {exc}\n"
            "Ensure terratorch is installed: pip install terratorch"
        ) from exc

    # Load checkpoint
    logger.info("Loading weights from %s on %s", weights_path, device)
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)

    # Checkpoints may be wrapped under 'state_dict'
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip Lightning prefix 'model.' if present
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[len("model."):] if k.startswith("model.") else k
        cleaned[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s …", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:5])

    model.to(device)
    model.eval()
    logger.info("Model ready on %s", device)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Run a forward pass and return a binary burn-scar mask.

    Parameters
    ----------
    model  : loaded Prithvi model
    tensor : (1, 6, H, W) float32 torch.Tensor — normalised
    device : torch.device

    Returns
    -------
    mask : np.ndarray  (H, W) uint8 — 0=background, 1=burn scar
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)

    # TerraTorch ModelOutput has an .output attribute; unwrap as needed
    if hasattr(output, "output"):
        logits = output.output
    elif isinstance(output, dict):
        logits = output.get("logits") or output.get("output") or next(iter(output.values()))
    elif isinstance(output, (list, tuple)):
        logits = output[0]
    else:
        logits = output   # plain tensor

    # logits: (1, 2, H, W) → argmax over class dim → (H, W)
    if logits.ndim == 4:
        pred = logits.argmax(dim=1).squeeze(0)
    elif logits.ndim == 3:
        pred = logits.squeeze(0) if logits.shape[0] == 1 else logits.argmax(dim=0)
    else:
        pred = logits

    return pred.cpu().numpy().astype(np.uint8)
