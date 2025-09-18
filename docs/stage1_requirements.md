# Stage-1 Compliance Requirements

## Overview
This document outlines the requirements for Stage-1 compliance in the vessel detection project. Beyond the current vessel detection capabilities, Stage-1 requires three main additional deliverables:

1. Correlation with AIS data
2. Path interpolation for vessel tracks
3. Standardized output formatting

## 1. Correlation with AIS

### Requirement
Match detected vessels from satellite imagery with AIS (Automatic Identification System) messages.

### Input Data
- **Vessel Detections**: Latitude, longitude, timestamp, and size from satellite imagery
- **AIS Messages**: MMSI (Maritime Mobile Service Identity), position reports, timestamp

### Required Output
CSV file with the following columns:
- `image_name`: Name of the satellite image
- `timestamp`: Time of detection
- `vessel_latitude`: Detected vessel latitude
- `vessel_longitude`: Detected vessel longitude
- `mmsi`: Matched AIS MMSI identifier

### Evaluation Metric
- **F1-score** for AIS correlation accuracy

## 2. Path Interpolation

### Requirement
For sparse AIS points, interpolate intermediate positions to create continuous vessel tracks.

### Input Data
- Matched AIS points (potentially sparse in time)

### Required Output
CSV file with the following columns:
- `timestamp`: Time of the interpolated position
- `path_id`: Unique identifier for the vessel track
- `point_id`: Sequential identifier for points along a path
- `point_latitude`: Interpolated latitude
- `point_longitude`: Interpolated longitude
- `speed_on_ground`: Interpolated speed
- `course_on_ground`: Interpolated course/heading

### Evaluation Metric
- **RMSE** (Root Mean Square Error) for interpolation accuracy

## 3. Output Formatting

### Detection Results
- **GeoJSON format** with vessel detections
- **Shapefile (.shp) format** with bounding boxes in latitude/longitude

### AIS Correlation and Interpolation
- CSV formats as specified in Appendix-B of the problem statement

## Current Gaps to Address

### 1. Data Pipeline Completeness
- Ensure proper ingestion of:
  - Sentinel Electro-Optical (EO) imagery
  - Sentinel Synthetic Aperture Radar (SAR) imagery
  - AIS CSV data in defined formats

### 2. Correlation Algorithm
- Implement a matching algorithm between detections and AIS tracks
- Likely approach: Nearest neighbor search in time and space
- Use vessel size as an additional filter for matching

### 3. Path Interpolation Algorithm
- Implement interpolation methods for AIS tracks:
  - Linear interpolation
  - Great-circle interpolation
  - Motion models (constant velocity, etc.)

### 4. Standardized Outputs
- Convert current detection table to required GeoJSON/shapefile formats
- Implement CSV output generation for correlation and interpolation results

## Evaluation Metrics Summary

| Component | Metric | Description |
|-----------|--------|-------------|
| Detection | AP50 | Average Precision at IoU 0.5 |
| AIS Correlation | F1-score | Harmonic mean of precision and recall |
| Path Interpolation | RMSE | Root Mean Square Error of interpolated positions |