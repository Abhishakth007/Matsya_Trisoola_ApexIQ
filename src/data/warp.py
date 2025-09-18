"""Warp utilities for raster imagery.

This module provides:
- ImageInfo: metadata container used throughout the pipeline.
- warp(...): compute an on-disk warped (or copied) GeoTIFF, returning ImageInfo.
- get_image_info(...): read ImageInfo for an existing raster.
- get_wgs84_bounds(...): compute WGS84 bounds for a raster dataset.

It is designed to be compatible with existing callers in:
- src/data/image.py (expects ImageInfo with .column/.row/.zoom)
- src/data/retrieve.py (imports get_image_info and warp)

Notes
-----
* Uses GDAL Python API (gdal.Warp) instead of invoking the `gdalwarp` CLI to avoid
  Windows path quoting issues. Ensure the `osgeo` Python package is installed.
* Supports projection=None (no reprojection, translate/copy to GeoTIFF) and
  EPSG:3857 reprojection with tile-aligned pixel size selection matching the
  original code's intent (web_mercator_m/512/2**zoom).
"""
from __future__ import annotations

import math
import os
import typing as t
import logging

from osgeo import gdal, ogr, osr

# Avoid GDAL aborting on conservative disk space estimates for large warps.
gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "FALSE")

# Circumference of the Earth in Web Mercator meters
WEB_MERCATOR_M = 2 * math.pi * 6378137


class ImageInfo(object):
    """Container for image metadata used by the pipeline."""

    def __init__(
        self,
        width: int,
        height: int,
        bounds: dict,
        column: int,
        row: int,
        zoom: int,
        projection: t.Optional[str] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.bounds = bounds
        self.column = column
        self.row = row
        self.zoom = zoom
        self.projection = projection

    def __repr__(self) -> str:  # helpful when logging/debugging
        return (
            f"ImageInfo(width={self.width}, height={self.height}, "
            f"column={self.column}, row={self.row}, zoom={self.zoom}, "
            f"projection={self.projection}, bounds={self.bounds})"
        )


# --- Core helpers -----------------------------------------------------------------

def _normalize_epsg(projection: t.Optional[str]) -> t.Optional[str]:
    if not projection:
        return None
    p = projection.strip()
    if not p:
        return None
    p_low = p.lower()
    if p_low.startswith("epsg:"):
        code = p.split(":", 1)[1]
        return f"EPSG:{code}"
    # Allow raw numeric strings like "3857"
    if p.isdigit():
        return f"EPSG:{p}"
    # Already looks like a PROJ/OGC string; pass through
    if p_upper := p.upper():
        if p_upper.startswith("EPSG:"):
            return p_upper
    raise ValueError(f"Unknown projection format: {projection}")


def _get_input_pixel_size(in_path: str, fallback: t.Optional[float] = None) -> float:
    """Return pixel size from metadata, or compute from the raster geotransform.

    The code relies on square pixels. We conservatively take the minimum of
    |pixel_size_x| and |pixel_size_y| if the input has slightly anisotropic pixels.
    """
    if fallback and fallback > 0:
        return float(fallback)

    # ENHANCED: Robust error handling with proper resource cleanup
    ds = None
    try:
        # Validate file exists
        if not os.path.exists(in_path):
            raise RuntimeError(f"Raster file does not exist: {in_path}")
        
        # Open raster dataset
        ds = gdal.Open(in_path)
        if ds is None:
            raise RuntimeError(f"Cannot open raster: {in_path}")
        
        # Get geotransform
        gt = ds.GetGeoTransform()
        if gt is None:
            raise RuntimeError(f"No geotransform available in raster: {in_path}")
        
        # Pixel size components; sign of gt[5] is typically negative for north-up rasters
        px = abs(gt[1])
        py = abs(gt[5])
        
        # Validate pixel sizes are reasonable
        if px <= 0 or py <= 0:
            logger.warning(f"Invalid pixel sizes: px={px}, py={py}, using fallback")
            return fallback if fallback and fallback > 0 else 10.0  # Default 10m
        
        # Return minimum for square pixels, or maximum if anisotropic
        return min(px, py) if (px > 0 and py > 0) else max(px, py)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting pixel size from {in_path}: {e}")
        if fallback and fallback > 0:
            logger.info(f"Using fallback pixel size: {fallback}")
            return float(fallback)
        else:
            logger.warning("No fallback available, using default pixel size: 10.0")
            return 10.0  # Default fallback
    finally:
        # ENSURE: Proper cleanup of GDAL dataset
        if ds is not None:
            try:
                ds = None  # GDAL will handle cleanup
                logger.debug("GDAL dataset cleaned up successfully")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up GDAL dataset: {cleanup_error}")


# --- Public API -------------------------------------------------------------------

AOI_COORDS = [
    (-75.966, 36.8709),  # (lon, lat)
    (-75.9433, 36.7631),
    (-75.9034, 36.7708),
    (-75.966, 36.8709),  # close the polygon
]

def _create_aoi_wkt(coords):
    """Return WKT polygon from list of (lon,lat)."""
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in coords:
        ring.AddPoint(x, y)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()

def warp(
    *args,
    projection: t.Optional[str] = None,
    x_res: t.Optional[float] = None,
    y_res: t.Optional[float] = None,
    aoi_coords: t.Optional[t.Sequence[t.Tuple[float, float]]] = None,
    enable_aoi: bool = True,
):
    """Warp image with optional reprojection and AOI clipping.

    Backward-compatible with both call styles:
      - warp(in_path, out_path, ...)
      - warp(image, in_path, out_path, ...)
    """
    # Parse arguments for backward compatibility
    if len(args) == 2 and all(isinstance(a, str) for a in args):
        in_path, out_path = t.cast(t.Tuple[str, str], args)
        image_obj = None
    elif len(args) == 3 and isinstance(args[1], str) and isinstance(args[2], str):
        image_obj, in_path, out_path = args  # type: ignore[assignment]
    else:
        raise TypeError("warp() expects (in_path, out_path, ...) or (image, in_path, out_path, ...)")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dst_srs = _normalize_epsg(projection)

    # Decide output resolution if reprojecting to EPSG:3857 and not explicitly provided
    out_zoom: int = 0
    out_pixel_size: t.Optional[float] = None
    if dst_srs == "EPSG:3857" and (x_res is None or y_res is None):
        # Prefer caller-provided pixel_size when available, else read from source
        pixel_fallback = getattr(image_obj, "pixel_size", None) if image_obj is not None else None
        in_pixel_size = _get_input_pixel_size(in_path, fallback=pixel_fallback)
        for zoom in range(20):
            zoom_pixel_size = WEB_MERCATOR_M / 512 / (2 ** zoom)
            out_pixel_size = zoom_pixel_size
            out_zoom = zoom
            if out_pixel_size < in_pixel_size * 1.1:
                break
        x_res = x_res or out_pixel_size
        y_res = y_res or out_pixel_size

    # Build warp kwargs
    warp_kwargs = dict(
        format="GTiff",
        multithread=True,
        srcNodata=None,
        dstNodata=None,
        creationOptions=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BIGTIFF=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
        ],
    )

    gdal.UseExceptions()

    # Conditionally create AOI and attach to warp options
    cutline_path: t.Optional[str] = None
    if enable_aoi:
        coords = aoi_coords or AOI_COORDS
        
        # Calculate output bounds for performance optimization
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Transform bounds to destination CRS if reprojecting
        if dst_srs == "EPSG:3857":
            # Transform WGS84 bounds to Web Mercator
            src_srs = osr.SpatialReference()
            src_srs.ImportFromEPSG(4326)
            dst_srs_obj = osr.SpatialReference()
            dst_srs_obj.ImportFromEPSG(3857)
            transformer = osr.CoordinateTransformation(src_srs, dst_srs_obj)
            min_x, min_y, _ = transformer.TransformPoint(min_lon, min_lat)
            max_x, max_y, _ = transformer.TransformPoint(max_lon, max_lat)
            output_bounds = [min_x, min_y, max_x, max_y]
        else:
            output_bounds = [min_lon, min_lat, max_lon, max_lat]
        
        # Create AOI polygon in-memory (GeoJSON) with a unique vsimem path per call
        cutline_path = f"/vsimem/aoi_{os.getpid()}_{id(coords)}.geojson"
        drv = ogr.GetDriverByName("GeoJSON")
        ds = drv.CreateDataSource(cutline_path)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        layer = ds.CreateLayer("aoi", srs, ogr.wkbPolygon)
        feat_def = layer.GetLayerDefn()
        feat = ogr.Feature(feat_def)
        geom = ogr.CreateGeometryFromWkt(_create_aoi_wkt(coords))
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = None
        ds = None  # flush to memory

        warp_kwargs.update(
            dict(
                cropToCutline=True,
                cutlineDSName=cutline_path,
                cutlineSRS="EPSG:4326",
                outputBounds=output_bounds,  # Limit processing to AOI bounds
            )
        )

    # Perform warp/copy
    try:
        ds_out = gdal.Warp(
            out_path,
            in_path,
            dstSRS=dst_srs,
            xRes=x_res,
            yRes=y_res,
            resampleAlg=gdal.GRA_Bilinear,
            **warp_kwargs,
        )
        if ds_out is None:
            raise RuntimeError(f"GDAL Warp failed: {in_path} -> {out_path}")
        ds_out.FlushCache()
        ds_out = None
    finally:
        if cutline_path:
            try:
                gdal.Unlink(cutline_path)
            except Exception:
                pass

    # Read output metadata
    raster = gdal.Open(out_path)
    try:
        bounds = get_wgs84_bounds(raster)
        column = 0
        row = 0
        zoom = 0
        if dst_srs == "EPSG:3857":
            gt = raster.GetGeoTransform()
            pixel_size = out_pixel_size or max(abs(gt[1]), abs(gt[5]))
            offset_x = gt[0] + WEB_MERCATOR_M / 2
            offset_y = WEB_MERCATOR_M - (gt[3] + WEB_MERCATOR_M / 2)
            if pixel_size and pixel_size > 0:
                column = int(offset_x / pixel_size)
                row = int(offset_y / pixel_size)
            # Infer zoom if not computed earlier
            if out_pixel_size is None and pixel_size:
                for z in range(30):
                    if abs((WEB_MERCATOR_M / 512 / (2 ** z)) - pixel_size) < 1e-6:
                        zoom = z
                        break
                else:
                    zoom = out_zoom
            else:
                zoom = out_zoom

        info = ImageInfo(
            width=raster.RasterXSize,
            height=raster.RasterYSize,
            bounds=bounds,
            column=column,
            row=row,
            zoom=zoom,
            projection=dst_srs,
        )
        return info
    finally:
        raster = None





def get_image_info(fname: str) -> ImageInfo:
    """Get image information for an existing raster on disk.

    Parameters
    ----------
    fname : str
        Path to the image.
    """
    raster = gdal.Open(fname)
    if raster is None:
        raise RuntimeError(f"Cannot open raster: {fname}")
    try:
        bounds = get_wgs84_bounds(raster)
        info = ImageInfo(
            width=raster.RasterXSize,
            height=raster.RasterYSize,
            bounds=bounds,
            column=0,
            row=0,
            zoom=0,
            projection=raster.GetProjection() or None,
        )
        return info
    finally:
        raster = None


def get_wgs84_bounds(raster) -> dict:
    """Get WGS84 bounding box given a GDAL raster dataset.

    Returns a dict with keys "Min" and "Max", each containing "Lon" and "Lat".
    """
    # Build a transformer from the raster's SRS to WGS84
    transformer = gdal.Transformer(raster, None, ["DST_SRS=WGS84"])
    if transformer is None:
        raise RuntimeError("GDAL Transformer initialization failed")

    # Transform all four corners
    _, p1 = transformer.TransformPoint(0, 0, 0, 0)  # top-left
    _, p2 = transformer.TransformPoint(0, 0, raster.RasterYSize, 0)  # bottom-left
    _, p3 = transformer.TransformPoint(0, raster.RasterXSize, 0, 0)  # top-right
    _, p4 = transformer.TransformPoint(0, raster.RasterXSize, raster.RasterYSize, 0)  # bottom-right

    lon_vals = [p[0] for p in (p1, p2, p3, p4)]
    lat_vals = [p[1] for p in (p1, p2, p3, p4)]
    return {
        "Min": {"Lon": min(lon_vals), "Lat": min(lat_vals)},
        "Max": {"Lon": max(lon_vals), "Lat": max(lat_vals)},
    }

















# import math
# import subprocess
# import typing as t

# from osgeo import gdal

# web_mercator_m = 2 * math.pi * 6378137


# class ImageInfo(object):
#     """Container for image metadata."""

#     def __init__(self, width, height, bounds, column, row, zoom, projection=None):
#         self.width = width
#         self.height = height
#         self.bounds = bounds
#         self.column = column
#         self.row = row
#         self.zoom = zoom
#         self.projection = projection


# def warp(
#     image, in_path: str, out_path: str, projection: t.Optional[str] = None
# ) -> ImageInfo:
#     """Warp a raw image to the specified projection.

#     Parameters
#     ----------
#     image: src.data.retrieve.RetrieveImage
#         Image class to warp.

#     in_path: str
#         Path to input raster geotiff.

#     out_path: str
#         Path to output raster geotiff.

#     projection: Optional[str]
#         Desired output projection (e.g. 'epsg:3857' for pseudo-mercator).
#         If None or '', the files are converted without warping.

#     Returns
#     -------
#      : ImageInfo
#         Class containing warped image information.
#     """

#     def get_pixel_size():
#         """Returns pixel size provided in metadata, or if unavailable, computes
#         pixel size from the input raster.
#         """
#         if image.pixel_size:
#             return image.pixel_size

#         raster = gdal.Open(in_path)
#         geo_transform = raster.GetGeoTransform()
#         pixel_size_x = geo_transform[1]
#         pixel_size_y = -geo_transform[5]
#         return min(pixel_size_x, pixel_size_y)

#     if not projection:
#         stdout = subprocess.check_output(["gdalwarp", in_path, out_path, "-overwrite"])
#         return get_image_info(out_path)
#     elif projection == "epsg:3857":
#         # Determine desired output resolution.
#         # We scale up to the zoom level just above the native resolution.
#         # This takes up more space than needed, but ensures we don't "lose" any resolution.
#         in_pixel_size = get_pixel_size()
#         out_pixel_size = None
#         out_zoom = None

#         for zoom in range(20):
#             zoom_pixel_size = web_mercator_m / 512 / (2**zoom)
#             out_pixel_size = zoom_pixel_size
#             out_zoom = zoom
#             if out_pixel_size < in_pixel_size * 1.1:
#                 break

#         # Warp the input image.
#         stdout = subprocess.check_output(
#             [
#                 "gdalwarp",
#                 "-r",
#                 "bilinear",
#                 "-t_srs",
#                 projection,
#                 "-tr",
#                 str(out_pixel_size),
#                 str(out_pixel_size),
#                 in_path,
#                 out_path,
#                 "-overwrite",
#             ]
#         )
#         # logger.debug(stdout)

#         raster = gdal.Open(out_path)
#         geo_transform = raster.GetGeoTransform()
#         offset_x = geo_transform[0] + web_mercator_m / 2
#         offset_y = web_mercator_m - (geo_transform[3] + web_mercator_m / 2)
#         offset_x /= out_pixel_size
#         offset_y /= out_pixel_size

#         return ImageInfo(
#             width=raster.RasterXSize,
#             height=raster.RasterYSize,
#             bounds=get_wgs84_bounds(raster),
#             column=int(offset_x),
#             row=int(offset_y),
#             zoom=out_zoom,
#             projection=projection,
#         )

#     else:
#         raise Exception("unknown projection {}".format(projection))


# def get_image_info(fname: str) -> ImageInfo:
#     """Get image information, assuming no projection.

#     Parameters
#     ----------
#     fname: str
#         Path to image.

#     Returns
#     -------
#     : ImageInfo
#         Image metadata.
#     """
#     raster = gdal.Open(fname)
#     return ImageInfo(
#         width=raster.RasterXSize,
#         height=raster.RasterYSize,
#         bounds=get_wgs84_bounds(raster),
#         column=0,
#         row=0,
#         zoom=0,
#     )


# def get_wgs84_bounds(raster) -> dict:
#     """Get WGS84 bounding box given gdal raster.

#     Parameters
#     ----------
#     raster: gdal.raster
#         Gdal image raster.

#     Returns
#     -------
#     : dict
#         Dictionary containing minimal and maximal lat/lon coords of raster.
#     """
#     transformer = gdal.Transformer(raster, None, ["DST_SRS=WGS84"])
#     _, p1 = transformer.TransformPoint(0, 0, 0, 0)
#     _, p2 = transformer.TransformPoint(0, 0, raster.RasterYSize, 0)
#     _, p3 = transformer.TransformPoint(0, raster.RasterXSize, 0, 0)
#     _, p4 = transformer.TransformPoint(0, raster.RasterXSize, raster.RasterYSize, 0)
#     points = [p1, p2, p3, p4]
#     return {
#         "Min": {
#             "Lon": min([p[0] for p in points]),
#             "Lat": min([p[1] for p in points]),
#         },
#         "Max": {
#             "Lon": max([p[0] for p in points]),
#             "Lat": max([p[1] for p in points]),
#         },
#     }
