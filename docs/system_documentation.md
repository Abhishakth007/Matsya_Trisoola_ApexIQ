# Sentinel Vessel Detection System — Comprehensive Documentation

## 1. Overview
The Sentinel Vessel Detection (SVD) system detects vessels in satellite imagery from the Sentinel-1 (SAR) and Sentinel-2 (optical) constellations and predicts per-vessel attributes (length, width, speed, heading bucket, fishing vessel probability). It supports large scenes at production scale and exposes a FastAPI endpoint for on-demand inference.

- **Imagery catalogs**: sentinel1 (SAR), sentinel2 (EO/IR)
- **Primary tasks**:
  - **Vessel detection (point-level)** using a Faster R-CNN–based architecture
  - **Attribute prediction** using a ResNet-based head (S1/S2 variants)
- **Key entry points**:
  - API: `src/main.py` (FastAPI, POST /detections)
  - Inference pipeline: `src/inference/pipeline.py`
  - Scene preparation: `src/data/image.py:prepare_scenes`


## 2. Architecture and Data Flow

### 2.1 High-level pipeline
1. **Preprocessing (prepare_scenes)**
   - Warps raw imagery to Web Mercator (EPSG:3857)
   - Aligns historical overlaps if provided
   - Verifies channel requirements, loads channels, and writes:
     - A base GeoTIFF (`*_base.tif`) for georeferencing
     - A concatenated array (`*_cat.npy`) with all channels selected for inference
2. **Detection + attributes (detect_vessels)**
   - Loads the preprocessed array (or uses in-memory `img_array` if provided)
   - Runs the detector with a sliding-window strategy (apply_model)
   - Applies non-maximum suppression (optional)
   - Runs the attribute predictor per detection crop
   - Transforms pixel coordinates back to the raw input space
   - Saves detection crops (optional) and a `predictions.csv`

### 2.2 Components
- **API Service (FastAPI)**: `src/main.py`
  - Request model: `SVDRequest`
  - Response model: `SVDResponse`
  - Orchestrates `prepare_scenes` and `detect_vessels`
- **Preprocessing**: `src/data/image.py`
  - `prepare_scenes` performs warp and alignment; outputs `.tif` and `.npy`
- **Inference**: `src/inference/pipeline.py`
  - `load_model` loads model from `cfg.json` + `best.pth`
  - `apply_model` performs sliding-window inference and postprocessing
  - `detect_vessels` coordinates the full inference workflow and I/O


## 3. Inputs and Supported Catalogs

### 3.1 Sentinel-1 (SAR)
- Product type: Level-1 GRD, Interferometric Wide Swath (IW)
- Required channels in `raw_path/<SCENE_ID>\measurement`: **VV**, **VH**
- Example scene ID: `S1A_IW_GRDH_... .SAFE`
- Pixel size assumption: 10 m native (warped, resampled to 0–255 uint8 per channel)

### 3.2 Sentinel-2 (Optical)
- Product type: L1C (MSI)
- Channel mapping (subset depends on model config):
  - **TCI** (RGB, 3 bands) → path abbreviation `TCI`
  - Example additional spectral bands supported by configs: `B08`, `B11`, `B12`
- Paths under `raw_path/<SCENE_ID>/GRANULE/*/IMG_DATA/*_{ABBREV}.jp2`

### 3.3 API Request Parameters (SVDRequest)
- **scene_id**: target scene directory name (e.g., S1A_... .SAFE)
- **raw_path**: folder containing the raw scene directory
- **output_dir**: folder where predictions and crops are saved
- **force_cpu**: bool, force CPU inference
- **historical1**, **historical2**: optional historical scene IDs for overlap features
- **window_size**: sliding window size (default 2048)
- **padding**: guard band inside each window (default 400)
- **overlap**: overlap allowed across windows in keep-bounds (default 20)
- **nms_thresh**: NMS suppression radius in pixels (default 10)
- **conf**: detection confidence threshold (default 0.9)
- **save_crops**: save per-detection crops (default True)
- **detector_batch_size**: batch size for detector (default 4)
- **postprocessor_batch_size**: attribute model batch size (default 32)
- **debug_mode**: extra logging
- **remove_clouds**: for S2, optional cloud filtering (currently unused in codepaths shown)


## 4. Scene Preparation (prepare_scenes)
File: `src/data/image.py`

### 4.1 Responsibilities
- Read model configs for detector and postprocessor to ensure shared base channels
- Build per-channel jobs:
  - Sentinel-1: collects `vh.tiff`, `vv.tiff` from `measurement/`
  - Sentinel-2: locates channels by JP2 patterns (e.g., TCI)
- Warp to EPSG:3857 using `src/data/warp.warp` (via GDAL) in parallel
- Align overlaps across scenes with subpixel search (max 32 px at quarter-res)
- Create a base GeoTIFF (`*_base.tif`) and concatenate warped arrays to `*_cat.npy`

### 4.2 Important behaviors
- Channel compatibility check: both detector and postprocessor must share the same underlying base channels (ignoring overlap channels)
- Image typed to uint8 [0–255] after clipping
- Automatic alignment uses dot-product search at reduced scale and rescales offsets


## 5. Inference Pipeline (detect_vessels, apply_model)
File: `src/inference/pipeline.py`

### 5.1 Model loading
- `load_model(model_dir, example, device)`
  - Reads `cfg.json` for architecture, channels, options
  - Loads `best.pth` weights
  - Supported architectures (from `src/models/__init__.py`):
    - `frcnn`, `frcnn_cmp2` (FasterRCNN variants)
    - `unet`
    - `resnet`
    - `custom`, `custom_separate_heads` (attribute heads)

### 5.2 Sliding-window detection (apply_model)
- Inputs: `img` tensor (C,H,W) as uint8 or float, window hyperparams
- Creates row/col offsets with keep-bounds per window:
  - Keeps detections within `[padding, window_size - padding]`, with relaxation by `overlap`
- Thresholds by `score >= conf`
- Transforms window-centered pixel coordinates to geodetic lat/lon via GDAL `Transformer(layer, None, ["DST_SRS=WGS84"])`
- Accumulates raw detections to a DataFrame with base fields:
  - `preprocess_row`, `preprocess_column`, `lon`, `lat`, `score`

### 5.3 Non-Maximum Suppression (NMS)
- Optional NMS via `nms(pred, distance_thresh)` using a grid index
- Removes duplicate detections closer than `nms_thresh` to a higher-score detection

### 5.4 Attribute prediction
- Batching: `postprocessor_batch_size` (minimum 1)
- For each detection:
  - Extracts a centered crop (default `crop_size=120`) from appropriate channels
  - Feeds to attribute model to obtain:
    - `vessel_length_m` (scaled by 100×)
    - `vessel_width_m` (scaled by 100×)
    - `heading_bucket_0..15` (softmax probabilities)
    - `vessel_speed_k` (knots)
    - `is_fishing_vessel` (class probability)

### 5.5 Coordinate back-mapping to input space
- Selects a suitable raw raster for back-transform:
  - S1: prioritize VH, then VV; if unavailable, deterministic sorted fallback
  - S2: uses TCI by default
- Builds a GDAL transform from output (warped) back to input raster
- Appends integer `column`, `row` in the input raster space

### 5.6 Crop saving and pixel metrics
- `save_detection_crops`
  - Extracts a square crop around detection center from the preprocessed image
  - Normalizes channels to uint8 using robust percentiles (2–98%)
  - Saves SAR channels (`vh`, `vv`) or S2 `tci` as PNGs
  - Computes `meters_per_pixel` using geodesic distances between crop corner lat/lons
  - Sets `orientation = 0` (north-up in Web Mercator)

### 5.7 Output
- Writes `predictions.csv` to `output_dir` with columns:
  - `preprocess_row`, `preprocess_column`, `lon`, `lat`, `score`
  - `vessel_length_m`, `vessel_width_m`
  - `heading_bucket_0..15`
  - `vessel_speed_k`, `is_fishing_vessel`
  - `column`, `row` (coordinates in the original input raster)
  - `orientation` (always 0)
  - `meters_per_pixel` (approximate)
  - `detect_id` (scene-based unique ID), `scene_id`
- If `save_crops=True`, corresponding `*_vh.png` and `*_vv.png` (S1) or `*_tci.png` (S2) are written


## 6. Models and Methodology

### 6.1 Detection models
- Sentinel-1 default: `frcnn_cmp2` (Faster R-CNN variant with customized backbone)
- Sentinel-2 default: `frcnn_cmp2` with a SwinV2 backbone (per model card)
- Sliding-window inference with confidence thresholding and NMS

### 6.2 Attribute models
- ResNet-50 backbone with task-specific heads (regression and classification)
- Predicts length, width (meters), speed (knots), and heading bucket distribution (16-way softmax); infers fishing probability

### 6.3 Evaluation metrics (from model cards)
- Detection: Precision, Recall, F1 (threshold curves)
- Attributes:
  - Length/Width: MAE and Average Percent Error Score
  - Speed: MAE (all) and MAE (moving-only), Average Percent Error Score
  - Heading: Accuracy, Average Degree Error Score (axis-corrected variant reported)

### 6.4 Reported results (validation)
- Sentinel-1 (artifacts: 3dff445 detector, c34aa37 attributes)
  - Precision: 0.836; Recall: 0.819; F1: 0.827
  - Length MAE: 27 m; Width MAE: 4.9 m; Speed MAE: 1.7 kt (2.37 kt moving-only)
  - Heading Accuracy: 27.8%; Additional heading score: ~0.57–0.74 (per model card)
- Sentinel-2 (artifacts: 15cddd5 detector, e609150 attributes)
  - Precision: 0.819; Recall: 0.790; F1: 0.804
  - Length MAE: 36 m; Width MAE: 5.6 m; Speed MAE: 1.4 kt (2.62 kt moving-only)
  - Heading Accuracy: 27.6%


## 7. Libraries and System Requirements

### 7.1 Core dependencies (versions from requirements.txt)
- torch >= 2.3; torchvision >= 0.18
- fastapi == 0.103.*; uvicorn >= 0.29
- gdal >= 3.8; rasterio >= 1.4; pyproj >= 3.6
- numpy >= 1.26; pandas >= 2.2; scikit-image >= 0.22; scikit-learn >= 1.5
- sentinelsat == 1.2.*; s2cloudless == 1.7.*
- lightgbm >= 4.5 (not directly used in inference path above)
- prometheus-client, pyfaktory, requests, urllib3, wandb

### 7.2 Runtime assumptions
- Python 3.8+ (up to 3.13 supported in repo docs)
- CUDA GPU optional; CPU supported (use `force_cpu` or device detection)
- GDAL available and compatible with Python version


## 8. API Usage

### 8.1 Endpoint
- `POST /detections`

### 8.2 Request body (JSON)
```json
{
  "scene_id": "S1A_IW_GRDH_... .SAFE",
  "output_dir": "d:/.../output",
  "raw_path": "d:/.../data",
  "force_cpu": false,
  "historical1": null,
  "historical2": null,
  "window_size": 2048,
  "padding": 400,
  "overlap": 20,
  "avoid": false,
  "nms_thresh": 10,
  "conf": 0.9,
  "save_crops": true,
  "detector_batch_size": 4,
  "postprocessor_batch_size": 32,
  "debug_mode": false,
  "remove_clouds": false
}
```

### 8.3 Response
```json
{ "status": "200" }
```


## 9. Outputs

### 9.1 predictions.csv columns
- `preprocess_row`, `preprocess_column`: detection centers in preprocessed space
- `lon`, `lat`: WGS84 location derived from output transform
- `score`: detector confidence
- `vessel_length_m`, `vessel_width_m`: attribute predictions (meters)
- `heading_bucket_0..15`: softmax probabilities by heading bucket
- `vessel_speed_k`: predicted speed in knots
- `is_fishing_vessel`: probability the object is a fishing vessel
- `column`, `row`: pixel coordinates back-projected to a selected raw raster (see §5.5)
- `orientation`: crop orientation (0, north-up)
- `meters_per_pixel`: approximate crop pixel size (meters/pixel)
- `detect_id`: unique per detection, `"<scene_id>_<index>"`
- `scene_id`: the scene ID

### 9.2 Crop images
- Sentinel-1: `{detect_id}_vh.png`, `{detect_id}_vv.png`
- Sentinel-2: `{detect_id}_tci.png`
- PNG crops saved as uint8 with robust per-channel normalization (2–98% percentiles)


## 10. Half-scene Test (performance-focused)
File: `test_half_scene.py`

### 10.1 Purpose
- Process only half of a Sentinel-1 scene to reduce time and demonstrate end-to-end outputs (predictions + crops)

### 10.2 Parameters used
- `window_size=512`, `padding=100`, `overlap=25`, `conf=0.15`
- Device: CPU
- `save_crops=True`
- `detector_batch_size=2`, `postprocessor_batch_size=2`

### 10.3 Results (from run on included scene)
- Output directory: `data/half_scene_test/`
- Files:
  - `predictions.csv`
  - 397 VH crops, 397 VV crops (one pair per detection)
- Aggregate metrics (parsed from output):
  - Total detections: 397
  - Score range: [0.150, 0.987]
  - Vessel length range (m): [15.2, 249.8]

Example first row (abridged):
```text
preprocess_row,preprocess_column,lon,lat,score,...,vessel_length_m,vessel_width_m,...,column,row,orientation,meters_per_pixel,detect_id,scene_id
129,4194,81.3233,17.4964,0.5476,...,56.12,12.93,...,24886,1,0,9.1626,S1A_..._0,S1A_...
```


## 11. Quality and Fixes Implemented
Summarized from `docs/root_cause_png_csv_issues.md` and current code:
- **PNG normalization**: crops are now normalized per-channel to uint8 (fixes dark/blank crops on float tensors)
- **Meters-per-pixel calculation**: uses float division to avoid zeros
- **S1 raster selection**: deterministic logic (prefer VH then VV) for back-mapping input coordinates
- **Postprocessor batch size**: respected via `postprocessor_batch_size` (batched loop in `apply_model`)

Open or optional improvements (non-blocking):
- **Crop size configurability** (currently out_crop_size=256 in `detect_vessels` when saving crops)
- **S2 back-mapping configurability** (defaults to TCI)


## 12. How to Run

### 12.1 Via API
1. Start server:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 5557
   ```
2. POST to `/detections` with the request JSON in §8.2

### 12.2 Via helper script
- `run_pipeline.py` shows how to invoke module entry with parameters (adjust paths for your environment)

### 12.3 Half-scene test
1. Ensure the raw `.SAFE` directory is under `data/`
2. Run:
   ```bash
   python test_half_scene.py
   ```
3. Inspect `data/half_scene_test/` for `predictions.csv` and PNG crops


## 13. File/Module Reference
- `src/main.py`: FastAPI app, request/response models, main endpoint
- `src/inference/pipeline.py`: detection workflow, NMS, attribute prediction, crop creation, CSV writing
- `src/data/image.py`: scene preparation, warping, alignment
- `src/models/__init__.py`: model architecture registry
- `docs/sentinel1_model_card.md`, `docs/sentinel2_model_card.md`: model details and validation metrics


## 14. Notes and Assumptions
- All imagery is warped to EPSG:3857; crops are north-up; `orientation=0`
- `meters_per_pixel` is approximate (geodesic-based, averaged across edges)
- For S1 back-mapping, VH/VV selection aims for consistency with base channels
- Historical overlaps are optional; when provided, the detector backbone may leverage temporal context


## 15. License and Contacts
- See repository LICENSE for terms
- Model Card Author: Mike Gartner (per model cards)