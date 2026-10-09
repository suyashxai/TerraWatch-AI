"""
model.py
--------
High-level orchestrator: accepts a raw file path, runs the full pipeline
(read -> preprocess -> inference -> postprocess) and returns structured results.

This is the single entry-point called by app.py so that Streamlit never
has to import rasterio, torch, or numpy directly.

RAM-saving measures (Streamlit Cloud):
- Model loaded once via st.cache_resource (survives reruns, shared across sessions).
- Raw raster array freed as soon as the tensor and RGB preview are built.
- Inference runs under torch.inference_mode() (less overhead than no_grad).
- Overlay is NOT stored in the returned dict; app.py recomputes it on demand.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

from backend.preprocessing import (
    read_geotiff,
    preprocess_for_inference,
    get_rgb_preview,
    validate_bands,
)
from backend.inference import load_model, run_inference, get_device
from backend.postprocessing import (
    save_mask_geotiff,
    build_overlay_image,
    calculate_statistics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model cache — loaded once per Streamlit server process, never reloaded
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Prithvi model weights…")
def _load_cached_model():
    """
    Load and return the Prithvi model.

    st.cache_resource guarantees this function runs exactly once per
    server process, even across Streamlit reruns and multiple users.
    The returned model object is shared (read-only at inference time).
    """
    logger.info("Loading Prithvi model (st.cache_resource — runs once per process)...")
    device = get_device()
    model = load_model(device)
    logger.info("Prithvi model ready on %s", device)
    return model, device


# ---------------------------------------------------------------------------
# Output directory helper
# ---------------------------------------------------------------------------

def _get_output_dir(requested: str) -> str:
    """
    Return a writable output directory.

    Tries the requested path first. Falls back to the system temp directory
    if the requested path cannot be created or written to (Streamlit Cloud).
    """
    import tempfile as _tempfile
    try:
        os.makedirs(requested, exist_ok=True)
        _probe = os.path.join(requested, ".write_test")
        with open(_probe, "w") as _f:
            _f.write("")
        os.remove(_probe)
        return requested
    except (OSError, PermissionError):
        logger.warning(
            "Output directory '%s' is not writable — falling back to system temp dir.",
            requested,
        )
        return _tempfile.gettempdir()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(tiff_path: str, output_dir: str = "outputs") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)
    """
    Full burn-scar detection pipeline.

    Parameters
    ----------
    tiff_path  : path to the uploaded 6-band HLS GeoTIFF
    output_dir : preferred directory for artefacts (mask GeoTIFF, overlay PNG).
                 Falls back to the system temp dir if not writable.

    Returns
    -------
    result : dict with keys
        rgb_preview      – np.ndarray (H, W, 3) uint8 — true-colour preview
        mask             – np.ndarray (H, W) uint8 — 0/1 burn-scar mask
        stats            – dict (burned_pixels, burned_area_km2, …)
        mask_path        – str, path to saved mask GeoTIFF
        overlay_path     – str, path to saved overlay PNG
        meta             – dict of image metadata
        inference_time_s – float, wall-clock seconds for the forward pass
        device_name      – str, "CPU" or "GPU (CUDA)"

    Note: the RGBA overlay array is NOT returned here. app.py recomputes it
    on demand from (rgb_preview, mask, opacity) to avoid storing a large
    redundant array in st.session_state.
    """
    output_dir = _get_output_dir(output_dir)

    # 1. Read raster from disk
    data, meta = read_geotiff(tiff_path)

    # 2. Validate band count
    validate_bands(data)

    # 3. Build RGB preview and normalised tensor, then free the raw raster.
    #    data is a (6, H, W) float32 array (~6 MB for 512×512).
    #    Releasing it here prevents it from competing for RAM with the model
    #    and the intermediate activation tensors during inference.
    rgb_preview = get_rgb_preview(data)
    tensor      = preprocess_for_inference(data)
    del data                    # free the 6-band raster array now

    # 4. Load model (cached) + run inference
    model, device = _load_cached_model()
    _t0 = time.perf_counter()
    mask = run_inference(model, tensor, device)
    inference_time_s = round(time.perf_counter() - _t0, 2)
    del tensor                  # free the normalised tensor

    # 5. Statistics and disk outputs
    stats = calculate_statistics(mask, meta)

    stem         = Path(tiff_path).stem
    mask_path    = os.path.join(output_dir, f"{stem}_mask.tif")
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")

    save_mask_geotiff(mask, meta, mask_path)

    # Write a default-opacity overlay to disk for the download button.
    # The interactive overlay rendered in the UI is computed separately
    # by app.py so we don't store a large RGBA array in session state.
    import imageio  # type: ignore
    _default_overlay = build_overlay_image(rgb_preview, mask, alpha=128)
    imageio.imwrite(overlay_path, _default_overlay)
    del _default_overlay        # not needed further in memory

    device_name = "GPU (CUDA)" if str(device) != "cpu" else "CPU"

    return {
        "rgb_preview":      rgb_preview,
        "mask":             mask,
        "stats":            stats,
        "mask_path":        mask_path,
        "overlay_path":     overlay_path,
        "meta":             meta,
        "inference_time_s": inference_time_s,
        "device_name":      device_name,
    }
