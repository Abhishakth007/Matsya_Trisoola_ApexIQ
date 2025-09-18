import os
import logging
from typing import Tuple, Optional

import numpy as np

try:
    from osgeo import gdal, osr
except Exception:
    gdal = None
    osr = None

from src.utils.snap_dimap_loader import load_snap_dimap
import numpy as _np
import cv2


logger = logging.getLogger("snap_preprocess")


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 [0,255] consistently with training/prep."""
    if image.dtype == np.uint8:
        return image
    # Normalize per-channel to [0,255]
    out = []
    for c in range(image.shape[0]):
        band = image[c]
        band_min = float(np.nanmin(band))
        band_max = float(np.nanmax(band))
        if band_max <= band_min:
            scaled = np.zeros_like(band, dtype=np.uint8)
        else:
            scaled = ((band - band_min) / (band_max - band_min) * 255.0).clip(0, 255).astype(np.uint8)
        out.append(scaled)
    return np.stack(out, axis=0)


def _normalize_percentile_uint8(band: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-based normalization to uint8 [0,255], ignoring zeros."""
    data = band.astype(_np.float32)
    mask = data > 0
    if not _np.any(mask):
        return _np.zeros_like(data, dtype=_np.uint8)
    lo, hi = _np.percentile(data[mask], [p_low, p_high])
    if hi <= lo:
        return _np.zeros_like(data, dtype=_np.uint8)
    scaled = (255.0 * (data - lo) / (hi - lo)).clip(0, 255)
    return scaled.astype(_np.uint8)


def _create_refined_overlap_channels(vv_u8: np.ndarray, vh_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Professor plan: SAR-specific overlaps from normalized VV/VH (uint8)."""
    vv_f = vv_u8.astype(_np.float32)
    vh_f = vh_u8.astype(_np.float32)

    # 1) Geometric mean
    overlap1 = _np.sqrt(vv_f * vh_f + 1e-6)
    overlap1 = overlap1.clip(0, 255).astype(_np.uint8)

    # 2) Polarization ratio (scaled)
    ratio = vv_f / (vh_f + 1e-6)
    overlap2 = (ratio * 50.0).clip(0, 255).astype(_np.uint8)

    # 3) Texture (multi-scale blur)
    overlap3 = cv2.GaussianBlur(vv_u8, (5, 5), 1.5)

    # 4) Edge (Sobel on VV)
    sob = cv2.Sobel(vv_f, cv2.CV_64F, 1, 1, ksize=3)
    overlap4 = _np.abs(sob).clip(0, 255).astype(_np.uint8)

    return overlap1, overlap2, overlap3, overlap4


def _write_base_geotiff(base_path: str, first_band: np.ndarray, geotransform, projection) -> None:
    if gdal is None:
        raise RuntimeError("GDAL not available to write base GeoTIFF")
    height, width = first_band.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(base_path, width, height, 1, gdal.GDT_Byte)
    if geotransform is not None:
        ds.SetGeoTransform(geotransform)
    if projection is not None:
        ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.WriteArray(first_band)
    band.FlushCache()
    ds.FlushCache()
    ds = None


def prepare_scenes_from_snap(
    input_path: str,
    scratch_path: str,
    detector_model_dir: str,
    postprocess_model_dir: str,
) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
    """
    Convert SNAP DIMAP or GeoTIFF into a 6-channel uint8 array and a valid base GeoTIFF.

    Returns (cat_array[C,H,W], base_path).
    """
    os.makedirs(scratch_path, exist_ok=True)

    # Load input
    img = None
    geotransform = None
    projection = None
    if input_path.lower().endswith(".dim"):
        img_array, meta = load_snap_dimap(input_path)
        img = img_array
        geo = meta.get("georeferencing", {})
        geotransform = geo.get("geotransform")
        projection = geo.get("projection")
    else:
        if gdal is None:
            raise RuntimeError("GDAL not available to read GeoTIFF")
        ds = gdal.Open(input_path)
        if ds is None:
            raise RuntimeError(f"Could not open {input_path}")
        bands = ds.RasterCount
        height, width = ds.RasterYSize, ds.RasterXSize
        img_list = []
        for i in range(1, bands + 1):
            img_list.append(ds.GetRasterBand(i).ReadAsArray())
        img = np.stack(img_list, axis=0).astype(np.float32)
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        ds = None

    # Identify VV and VH by heuristic: prefer names if available; else first two
    vv = None
    vh = None
    if img.shape[0] >= 2:
        vv = img[0]
        vh = img[1]
    else:
        raise ValueError("SNAP input must have at least 2 bands (VV,VH)")

    # Normalize VV/VH per-scene to match training distribution (1–99 percentiles)
    vv_u8 = _normalize_percentile_uint8(vv, 1.0, 99.0)
    vh_u8 = _normalize_percentile_uint8(vh, 1.0, 99.0)

    # Refined overlap channels
    ov1, ov2, ov3, ov4 = _create_refined_overlap_channels(vv_u8, vh_u8)

    cat = np.stack([vv_u8, vh_u8, ov1, ov2, ov3, ov4], axis=0).astype(np.uint8)

    # Write base geotiff using VV as base
    base_path = os.path.join(scratch_path, "snap_base.tif")
    try:
        _write_base_geotiff(base_path, vv_u8, geotransform, projection)
    except Exception as e:
        logger.warning(f"Failed to write base GeoTIFF with georeferencing: {e}. Writing without geo.")
        _write_base_geotiff(base_path, vv_u8, None, None)

    # Build simple water mask for SNAP-masked data: water if VV>0 or VH>0
    water_mask = ((vv_u8 > 0) | (vh_u8 > 0)).astype(np.uint8)

    return cat, base_path, water_mask


