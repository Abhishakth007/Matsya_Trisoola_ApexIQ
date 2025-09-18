import datetime
import glob
import json
import logging
import multiprocessing
import os
import sys
import typing as t
import gc
import psutil

import numpy as np
import skimage.io
import torch
from osgeo import gdal

# Fix GDAL warnings by explicitly setting exception handling
gdal.UseExceptions()  # Enable GDAL exceptions

from src.data.retrieve import RetrieveImage
from src.data.warp import warp
from src.utils.channel_manager import ChannelManager
from src.utils.gdal_manager import GDALResourceManager, robust_gdal_operation
from src.utils.memory_manager import get_memory_manager, memory_context, log_memory

# Memory monitoring utility
def log_memory_usage(stage=""):
    """Log current memory usage for debugging performance issues."""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage {stage}: {memory_mb:.1f} MB")
        return memory_mb
    except Exception:
        pass  # Don't fail if psutil is not available

def force_memory_cleanup():
    """Force garbage collection and memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Configure logger
logger = logging.getLogger("prepare_scenes")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s (%(levelname)s): %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

SENTINEL_1_REQUIRED_CHANNEL_TYPES = ["vh", "vv"]

# Map Sentinel-2 channels according to DB naming convention
# to raw imagery path naming convention, and channel count.
# Unlike Sentinel-1, actual channels used by model are specified
# via config that can be altered at runtime.
SENTINEL_2_CHANNEL_MAPPING = {"tci": {"path_abbrev": "TCI", "count": 3}, "b08": {"path_abbrev": "B08", "count": 1},
                              "b11": {"path_abbrev": "B11", "count": 1}, "b12": {"path_abbrev": "B12", "count": 1}}
SUPPORTED_IMAGERY_CATALOGS = ["sentinel1", "sentinel2"]


class InvalidDataError(Exception):
    pass


class InvalidConfigError(Exception):
    pass


class Channels(object):
    def __init__(self, channels):
        """
        Given a (JSON-decoded) list of fChannel, creates a Channels object.
        """
        self.channels = channels

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        return self.channels[idx]
    
    def count(self):
        """Return the total number of channels across all channel types."""
        return sum([channel["Count"] for channel in self.channels])

    def flatten(self):
        """
        For example, [tci, b5] -> ['tci-0', 'tci-1', 'tci-2', 'b5']
        """
        flat_list = []
        for channel in self.channels:
            if channel["Count"] > 1:
                for i in range(channel["Count"]):
                    flat_list.append("{}-{}".format(channel["Name"], i))
            else:
                flat_list.append(channel["Name"])
        return flat_list

    def with_ranges(self):
        l = []
        cur_idx = 0
        for channel in self.channels:
            l.append((channel, (cur_idx, cur_idx + channel["Count"])))
            cur_idx += channel["Count"]
        return l


def warp_func(job: list):
    """Perform a single image warp job.

    Parameters
    ----------
    job: list
        List specifying job params.

    Returns
    -------
    : ImageInfo
        Class containing warped image metadata.
    """
    retrieve_image, scene_id, channel_name, src_path, dst_path, aoi_coords = job
    # Convert AOI coordinates format if provided
    aoi_tuples = None
    if aoi_coords is not None:
        aoi_tuples = [(coord[0], coord[1]) for coord in aoi_coords]
    
    return warp(
        src_path, 
        dst_path, 
        projection="epsg:4326",
        aoi_coords=aoi_tuples,
        enable_aoi=aoi_coords is not None
    )



def prepare_scenes(
    raw_path: str,
    scratch_path: str,
    scene_id: str,
    historical1: t.Optional[str],
    historical2: t.Optional[str],
    catalog: str,
    cat_path: str,
    base_path: str,
    device: torch.device,
    detector_model_dir: str,
    postprocess_model_dir: str,
    aoi_coords: t.Optional[t.List[t.List[float]]] = None
) -> None:
    """
    Extract and warp scenes, then save numpy array with scene info.
    
    MAJOR PERFORMANCE OPTIMIZATIONS IMPLEMENTED:
    1. Skip expensive alignment for single images
    2. Memory monitoring and cleanup
    3. Optimized alignment algorithm
    4. Reduced tensor operations
    5. Efficient GDAL configuration
    """
    # Ensure logger is defined in all code paths (avoid NameError from accidental shadowing)
    try:
        _ = logger  # type: ignore[name-defined]
    except Exception:
        import logging as _logging
        globals()["logger"] = _logging.getLogger("prepare_scenes")

    logger.info(f"🚀 Starting optimized scene preparation for {scene_id}")
    
    # Initialize memory manager
    memory_manager = get_memory_manager()
    log_memory("start")
    
    # Check if we have historical images (main performance bottleneck)
    has_historical = historical1 is not None or historical2 is not None
    if not has_historical:
        logger.info("✅ No historical images - using fast single-image processing path")
    
    # Create scratch directory if it doesn't exist
    os.makedirs(scratch_path, exist_ok=True)
    
    # Load detector and postprocessor config
    with open(os.path.join(detector_model_dir, "cfg.json"), "r") as f:
        detector_cfg = json.load(f)
    with open(os.path.join(postprocess_model_dir, "cfg.json"), "r") as f:
        postprocess_cfg = json.load(f)

    # Verify channel requirements align (ignoring overlap channels),
    # since the models will share the same pre-processed base imagery
    postprocess_channels = set(ch["Name"] for ch in postprocess_cfg["Channels"] if "overlap" not in ch["Name"])
    detector_channels = set(ch["Name"] for ch in detector_cfg["Channels"] if "overlap" not in ch["Name"])
    if postprocess_channels != detector_channels:
        raise InvalidConfigError("Detector and postprocessor models are required to use the"
                                 f"same underlying channels.\n You passed"
                                 f" detector_channels={detector_channels}\n"
                                 f"postprocessor_channels={postprocess_channels}")

    # Warp the scenes, in parallel.
    # Each job is (retrieve_image, scene_id, channel_name, src_path, dst_path)
    retrieve_images = []
    jobs = []
    scene_ids = [scene_id]
    if historical1:
        scene_ids.append(historical1)
    if historical2:
        scene_ids.append(historical2)
    for scene_id in scene_ids:
        scene_channels = []
        if catalog == "sentinel1":
            measurement_path = os.path.join(raw_path, scene_id, "measurement")
            # Preflight: ensure measurement directory exists for GRD SAFE
            if not os.path.isdir(measurement_path):
                raise InvalidDataError(
                    "Sentinel-1 SAFE is missing the required 'measurement/' directory with VV/VH rasters.\n"
                    f"Checked path: {measurement_path}\n"
                    "Fix options: (1) Use a complete SAFE (e.g., data/<SAFE>.SAFE), "
                    "(2) Re-extract original ESA ZIP with 'measurement/' present, "
                    "(3) Preprocess with ESA SNAP GPT to GeoTIFF and point the pipeline to that."
                )
            # FIXED: Robust filename parsing that handles different filename formats
            fnames = {}
            for fname in os.listdir(measurement_path):
                if fname.endswith('.tiff') or fname.endswith('.tif'):
                    # Try to extract polarization from filename
                    if 'vh' in fname.lower():
                        fnames['vh'] = fname
                    elif 'vv' in fname.lower():
                        fnames['vv'] = fname
            
            # Fallback: if no polarization found, try the original parsing method
            if not fnames:
                try:
                    fnames = {
                        fname.split("-")[3]: fname for fname in os.listdir(measurement_path)
                        if len(fname.split("-")) > 3
                    }
                except (IndexError, KeyError):
                    pass
            if all(key in fnames for key in SENTINEL_1_REQUIRED_CHANNEL_TYPES):
                scene_channels.append(
                    {
                        "Name": "vh",
                        "Path": os.path.join(measurement_path, fnames["vh"]),
                        "Count": 1,
                    }
                )
                scene_channels.append(
                    {
                        "Name": "vv",
                        "Path": os.path.join(measurement_path, fnames["vv"]),
                        "Count": 1,
                    }
                )
            else:
                raise InvalidDataError(
                    f"Raw Sentinel-1 data must contain polarization channels={SENTINEL_1_REQUIRED_CHANNEL_TYPES}.\n"
                    f"Found: {fnames}"
                )
        elif catalog == "sentinel2":
            # Channels of interest for model, as specified in model cfg
            # Length rules out overlap channels from cfg, which are handled separately here
            sentinel_2_cois = [x["Name"] for x in detector_cfg["Channels"] if len(x["Name"]) == 3]
            sentinel_2_coi_map = dict((k, SENTINEL_2_CHANNEL_MAPPING[k])
                                      for k in sentinel_2_cois if k in SENTINEL_2_CHANNEL_MAPPING)
            for channel, val in sentinel_2_coi_map.items():
                path_abbrev = val["path_abbrev"]
                count = val["count"]
                path_pattern = os.path.join(raw_path, scene_id, f"GRANULE/*/IMG_DATA/*_{path_abbrev}.jp2")
                paths = glob.glob(path_pattern)
                if len(paths) == 1:
                    path = paths[0]
                    scene_channels.append(
                        {
                            "Name": channel,
                            "Path": path,
                            "Count": count
                        }
                    )
                else:
                    raise InvalidDataError(
                        f"Raw Sentinel-2 data must be of L1C product type, and contain channel={channel}.\n"
                        f"Did not find a unique path using the pattern: {path_pattern}"
                    )

        else:
            raise ValueError(
                f"You specified imagery catalog={catalog}.\n"
                f"The only supported catalogs are: {SUPPORTED_IMAGERY_CATALOGS}"
            )

        retrieve_image = RetrieveImage(
            uuid="x",
            name=scene_id,
            time=datetime.datetime.now(),
            format="x",
            channels=scene_channels,
            pixel_size=10,
        )
        retrieve_image.job_ids = []

        for ch in scene_channels:
            retrieve_image.job_ids.append(len(jobs))

            if len(jobs) == 0:
                dst_path = base_path
            else:
                dst_path = os.path.join(
                    scratch_path, scene_id + "_" + ch["Name"] + ".tif"
                )

            jobs.append(
                [
                    retrieve_image,
                    scene_id,
                    ch["Name"],
                    ch["Path"],
                    dst_path,
                    aoi_coords,
                ]
            )

        retrieve_images.append(retrieve_image)

    # Optimize GDAL configuration for better performance
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_CACHEMAX", "1024")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    
    # Use fewer processes since GDAL is already multithreaded
    # This prevents thread oversubscription on Windows
    num_workers = min(2, len(jobs), multiprocessing.cpu_count() // 2)
    
    if num_workers > 1 and len(jobs) > 1:
        p = multiprocessing.Pool(num_workers)
        image_infos = p.map(warp_func, jobs)
        p.close()
        p.join()
    else:
        # Process sequentially for small job counts
        image_infos = [warp_func(job) for job in jobs]

    log_memory("after GDAL processing")

    first_info = None
    ims = []
    
    # OPTIMIZATION: Process images with memory management
    for retrieve_image in retrieve_images:
        overlap_offset = (0, 0)
        for ch_idx, ch in enumerate(retrieve_image.channels):
            job_id = retrieve_image.job_ids[ch_idx]
            job = jobs[job_id]
            image_info = image_infos[job_id]
            _, _, _, _, tmp_path, _ = job

            # OPTIMIZATION: Load image more efficiently
            im = skimage.io.imread(tmp_path)
            im = np.clip(im, 0, 255).astype(np.uint8)
            if len(im.shape) == 2:
                im = im[None, :, :]
            else:
                im = im.transpose(2, 0, 1)
            
            # OPTIMIZATION: Convert to tensor only when needed
            if not first_info:
                first_info = image_info
                ims.append(im)  # Keep as numpy array initially
                continue

            # OPTIMIZATION: Skip expensive alignment for single images
            if not has_historical:
                logger.info("⏭️ Skipping alignment for single image processing")
                ims.append(im)
                continue

            # Align later images with the first one.
            left = image_info.column - first_info.column
            top = image_info.row - first_info.row

            # OPTIMIZATION: Only align VH or TCI channels (most important for vessel detection)
            if "vh" in ch["Name"].lower() or "tci" in ch["Name"].lower():
                base_padded = ims[0][0, :, :]
                other_padded = im[0, :, :]
                if top < 0:
                    other_padded = other_padded[-top:, :]
                else:
                    base_padded = base_padded[top:, :]
                if left < 0:
                    other_padded = other_padded[:, -left:]
                else:
                    base_padded = base_padded[:, left:]

                if other_padded.shape[0] > base_padded.shape[0]:
                    other_padded = other_padded[0: base_padded.shape[0], :]
                else:
                    base_padded = base_padded[0: other_padded.shape[0], :]
                if other_padded.shape[1] > base_padded.shape[1]:
                    other_padded = other_padded[:, 0: base_padded.shape[1]]
                else:
                    base_padded = base_padded[:, 0: other_padded.shape[1]]

                logger.info(f"🔄 Aligning {ch['Name']} channel for {retrieve_image.name}")
                
                # OPTIMIZATION: Use faster alignment algorithm
                overlap_offset = fast_image_alignment(base_padded, other_padded, top, left, device)
                
                logger.info(f"✅ Alignment complete for {ch['Name']}: offset {overlap_offset}")
                
                # Clean up tensors immediately
                del base_padded, other_padded
                memory_manager.force_memory_cleanup()

            left += overlap_offset[0]
            top += overlap_offset[1]

            if top < 0:
                im = im[:, -top:, :]
            else:
                im = torch.nn.functional.pad(im, (0, 0, top, 0))
            if left < 0:
                im = im[:, :, -left:]
            else:
                im = torch.nn.functional.pad(im, (left, 0, 0, 0))

            # Crop to size if needed.
            if im.shape[1] > first_info.height:
                im = im[:, 0: first_info.height, :]
            elif im.shape[1] < first_info.height:
                im = torch.nn.functional.pad(
                    im, (0, 0, 0, first_info.height - im.shape[1])
                )
            if im.shape[2] > first_info.width:
                im = im[:, :, 0: first_info.width]
            elif im.shape[2] < first_info.width:
                im = torch.nn.functional.pad(
                    im, (0, first_info.width - im.shape[2], 0, 0)
                )

            ims.append(im)
            
            # OPTIMIZATION: Force memory cleanup after each image
            if len(ims) % 3 == 0:  # Cleanup every 3 images
                memory_manager.force_memory_cleanup()

    log_memory("after image processing")

    # ENHANCED: Use ChannelManager for intelligent channel creation
    if not has_historical and catalog == "sentinel1" and len(ims) == 2:
        logger.info("🔄 No historical data available - using ChannelManager for intelligent channel creation")
        
        try:
            # Initialize ChannelManager with detector configuration
            channel_manager = ChannelManager(detector_cfg)
            
            # Create synthetic overlap channels
            synthetic_channels = channel_manager.create_channels(ims, catalog=catalog)
            
            # Validate channel count
            if channel_manager.validate_channel_count(synthetic_channels):
                logger.info(f"✅ ChannelManager created {len(synthetic_channels)} channels successfully")
                ims = synthetic_channels
            else:
                logger.warning("⚠️ ChannelManager validation failed, using fallback duplication")
                # Fallback to simple duplication
                base_vh = ims[0]  # VH channel
                base_vv = ims[1]  # VV channel
                ims.extend([base_vh, base_vv, base_vh, base_vv])  # Add 4 overlap channels
                logger.info(f"✅ Extended from 2 to 6 channels using fallback method")
        except Exception as e:
            logger.warning(f"⚠️ ChannelManager failed: {e}, using fallback duplication")
            # Fallback to simple duplication
            base_vh = ims[0]  # VH channel
            base_vv = ims[1]  # VV channel
            ims.extend([base_vh, base_vv, base_vh, base_vv])  # Add 4 overlap channels
            logger.info(f"✅ Extended from 2 to 6 channels using fallback method")
    
    # OPTIMIZATION: Concatenate efficiently
    logger.info(f"🔗 Concatenating {len(ims)} image channels")
    im = np.concatenate(ims, axis=0)
    
    # Clean up individual images immediately
    del ims
    memory_manager.force_memory_cleanup()

    log_memory("after concatenation")

    # Save concatenated image
    if cat_path:
        logger.info(f"💾 Saving concatenated image to {cat_path}")
        np.save(cat_path, im)
        logger.info(f"✅ Saved {im.shape} image to {cat_path}")

    log_memory("final")
    logger.info("🎉 Scene preparation completed successfully!")
    
    return im


def fast_image_alignment(base_img, other_img, top_offset, left_offset, device):
    """
    Optimized image alignment algorithm with reduced computational complexity.
    
    Args:
        base_img: Base image tensor (H, W)
        other_img: Image to align tensor (H, W)
        top_offset: Initial top offset
        left_offset: Initial top offset
        device: Device to use for computation
    
    Returns:
        tuple: (left_offset, top_offset) for alignment
    """
    # FIXED: Add input validation
    try:
        if base_img is None or other_img is None:
            logger.warning("Invalid input images for alignment")
            return (0, 0)
        
        if not isinstance(top_offset, (int, float)) or not isinstance(left_offset, (int, float)):
            logger.warning("Invalid offset parameters for alignment")
            return (0, 0)
            
        if device is None:
            logger.warning("Device not specified for alignment")
            device = torch.device("cpu")
            
    except Exception as e:
        logger.warning(f"Input validation failed for alignment: {e}")
        return (0, 0)
    
    # OPTIMIZATION: Use smaller search window for speed
    max_offset = 8  # Reduced from 16
    step_size = 2   # Reduced from 4
    
    # Apply initial offset
    if top_offset < 0:
        other_img = other_img[-top_offset:, :]
    else:
        base_img = base_img[top_offset:, :]
    if left_offset < 0:
        other_img = other_img[:, -left_offset:]
    else:
        base_img = base_img[:, left_offset:]

    # Ensure same dimensions
    min_h = min(base_img.shape[0], other_img.shape[0])
    min_w = min(base_img.shape[1], other_img.shape[1])
    
    base_crop = base_img[:min_h, :min_w]
    other_crop = other_img[:min_h, :min_w]
    
    # OPTIMIZATION: Use downsampling for faster correlation
    scale_factor = 4
    base_small = base_crop[::scale_factor, ::scale_factor]
    other_small = other_crop[::scale_factor, ::scale_factor]
    
    # Normalize for correlation
    base_norm = (base_small - base_small.mean()) / (base_small.std() + 1e-8)
    other_norm = (other_small - other_small.mean()) / (other_small.std() + 1e-8)
    
    # OPTIMIZATION: Reduced search space
    max_offset_small = max_offset // scale_factor
    
    best_offset = (0, 0)
    best_score = float('-inf')
    
    # OPTIMIZATION: Use larger step size for faster search
    for top_offset in range(-max_offset_small, max_offset_small + 1, step_size):
        for left_offset in range(-max_offset_small, max_offset_small + 1, step_size):
            # Calculate valid overlap region
            y_start = max(0, top_offset)
            y_end = min(base_small.shape[0], base_small.shape[0] + top_offset)
            x_start = max(0, left_offset)
            x_end = min(base_small.shape[1], base_small.shape[1] + left_offset)
            
            if y_end - y_start < 5 or x_end - x_start < 5:  # Skip tiny overlaps
                continue
                
            base_region = base_norm[y_start:y_end, x_start:x_end]
            other_y_start = y_start - top_offset
            other_y_end = y_end - top_offset
            other_x_start = x_start - left_offset
            other_x_end = x_end - left_offset
            
            if (other_y_start < 0 or other_y_end > other_small.shape[0] or 
                other_x_start < 0 or other_x_end > other_small.shape[1]):
                continue
                
            other_region = other_norm[other_y_start:other_y_end, other_x_start:other_x_end]
            
            if base_region.shape != other_region.shape:
                continue
                
            # Correlation score
            score = torch.mean(base_region * other_region)
            if score > best_score:
                best_offset = (left_offset, top_offset)
                best_score = score

    # Scale back to original resolution
    final_offset = (
        scale_factor * best_offset[0],
        scale_factor * best_offset[1],
    )
    
    return final_offset
