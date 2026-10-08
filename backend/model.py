"""
model.py
--------
High-level orchestrator: accepts a raw file path, runs the full pipeline
(read -> preprocess -> inference -> postprocess) and returns structured results.

This is the single entry-point called by app.py so that Streamlit never
has to import rasterio, torch, or numpy directly.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.preprocessing import (
    read_geotiff,
    preprocess_for_inference,
    get_rgb_preview,
    validate_bands,
)
from backend.inference import load_model, run_inference, get_device, weights_available
from backend.postprocessing import (
    save_mask_geotiff,
    build_overlay_image,
    calculate_statistics,
)

logger = logging.getLogger(__name__)

# Module-level model cache so that the expensive load happens once per session.
_model_cache: Dict[str, Any] = {}


def _get_model():
    if "model" not in _model_cache:
        logger.info("Loading Prithvi model for the first time (this may take a moment)...")
        device = get_device()
        model = load_model(device)
        _model_cache["model"] = model
        _model_cache["device"] = device
    return _model_cache["model"], _model_cache["device"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(tiff_path: str, output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Full burn-scar detection pipeline.

    Parameters
    ----------
    tiff_path  : path to the uploaded 6-band HLS GeoTIFF
    output_dir : directory where artefacts (mask, overlay) are written

    Returns
    -------
    result : dict with keys
        rgb_preview      – np.ndarray (H, W, 3) uint8 — true-colour preview
        mask             – np.ndarray (H, W) uint8 — 0/1 burn-scar mask
        overlay          – np.ndarray (H, W, 4) uint8 — RGBA overlay
        stats            – dict (burned_pixels, burned_area_km2, …)
        mask_path        – str, path to saved mask GeoTIFF
        overlay_path     – str, path to saved overlay PNG
        meta             – dict of image metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read
    data, meta = read_geotiff(tiff_path)

    # 2. Validate
    validate_bands(data)

    # 3. Preprocess
    tensor = preprocess_for_inference(data)

    # 4. Load model + run inference (measure wall-clock time)
    model, device = _get_model()
    _t0 = time.perf_counter()
    mask = run_inference(model, tensor, device)
    inference_time_s = round(time.perf_counter() - _t0, 2)

    # 5. Postprocess
    rgb_preview = get_rgb_preview(data)
    stats = calculate_statistics(mask, meta)

    stem = Path(tiff_path).stem
    mask_path = os.path.join(output_dir, f"{stem}_mask.tif")
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")

    save_mask_geotiff(mask, meta, mask_path)
    overlay = build_overlay_image(rgb_preview, mask, alpha=128)  # default 50 %

    import imageio  # type: ignore
    imageio.imwrite(overlay_path, overlay)

    import torch
    device_name = "GPU (CUDA)" if str(device) != "cpu" else "CPU"

    return {
        "rgb_preview":      rgb_preview,
        "mask":             mask,
        "overlay":          overlay,
        "stats":            stats,
        "mask_path":        mask_path,
        "overlay_path":     overlay_path,
        "meta":             meta,
        "inference_time_s": inference_time_s,
        "device_name":      device_name,
    }
