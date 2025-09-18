# ACTUAL PARAMETERS USED IN VESSEL DETECTION PIPELINE

## ✅ Parameters Actually Implemented and Used:

### 1. **confidence_threshold** (float)
- **Used in**: `ProcessingConfig.confidence_threshold`
- **Purpose**: Minimum confidence score for detections
- **Current value**: 0.85 (TOO HIGH - causing under-detection)
- **Documentation example**: 0.15 (found 397 detections)
- **Range**: 0.0 - 1.0

### 2. **nms_threshold** (float) 
- **Used in**: `ProcessingConfig.nms_threshold`
- **Purpose**: IoU threshold for Non-Maximum Suppression
- **Current value**: 0.15 (IoU-based)
- **Documentation example**: 10px (converted to ~0.1 IoU)
- **Range**: 0.0 - 1.0

### 3. **window_size** (int)
- **Used in**: `ProcessingConfig.window_size`
- **Purpose**: Size of processing windows for tiled inference
- **Current value**: 1024
- **Documentation examples**: 512 (successful), 2048 (default), 800 (training)
- **Range**: 400 - 2048

### 4. **overlap** (int)
- **Used in**: `ProcessingConfig.overlap`
- **Purpose**: Overlap between adjacent processing windows
- **Current value**: 128
- **Documentation examples**: 20 (default), 25 (successful), 200 (25% of 800)
- **Range**: 20 - 200

## ❌ Parameters NOT Used (Just Sitting in Config):

### 1. **water_threshold** (float)
- **Status**: ❌ NOT IMPLEMENTED
- **Location**: Only in config file `main.windowing.water_threshold`
- **Reality**: No code actually uses this parameter
- **Value**: 0.1 (meaningless)

### 2. **edge_threshold** (float)
- **Status**: ❌ NOT IMPLEMENTED  
- **Location**: Only in config file `main.windowing.edge_threshold`
- **Reality**: No code actually uses this parameter
- **Value**: 0.15 (meaningless)

### 3. **padding** (int)
- **Status**: ❌ NOT IMPLEMENTED in robust pipeline
- **Location**: Only in config file `main.default_padding`
- **Reality**: Robust pipeline doesn't use padding parameter
- **Value**: 400 (meaningless)

## 🎯 Key Finding:

**The main issue is the confidence_threshold = 0.85 is WAY TOO HIGH!**

- **Current**: 0.85 → ~4 detections
- **Documentation success**: 0.15 → 397 detections  
- **Expected improvement**: 50-100x more detections

## 📊 Recommended Testing Focus:

1. **confidence_threshold**: 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5
2. **nms_threshold**: 0.1, 0.15, 0.2, 0.25, 0.3, 0.4
3. **window_size**: 400, 512, 800, 1024, 1536, 2048
4. **overlap**: 50, 100, 150, 200 (proportional to window_size)

**Ignore**: water_threshold, edge_threshold, padding (not implemented)
