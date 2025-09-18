# Vessel Correlation and Path Interpolation Methodology

## 1. AIS Correlation Methodology

### Overview
The correlation process matches vessel detections from satellite imagery (Sentinel-1 SAR and Sentinel-2 EO/IR) with AIS (Automatic Identification System) messages to identify vessels and track their movements.

### Algorithm Steps

#### 1.1 Data Preparation
- **Satellite Detections**:
  - Extract vessel positions (latitude, longitude)
  - Record detection timestamp
  - Estimate vessel size (length, width) from bounding boxes
  - Calculate detection confidence score

- **AIS Messages**:
  - Parse AIS CSV data
  - Extract MMSI, timestamp, position, speed, course
  - Filter for messages within the temporal window of satellite imagery
  - Convert all coordinates to a consistent coordinate system

#### 1.2 Spatial-Temporal Matching

1. **Temporal Filtering**:
   - For each satellite detection, create a temporal window (e.g., ±30 minutes)
   - Filter AIS messages that fall within this temporal window

2. **Spatial Matching**:
   - For temporally filtered AIS messages, calculate spatial distance to detection
   - Use Haversine formula for accurate distance calculation on Earth's surface
   - Create candidate matches based on distance threshold

3. **Multi-Attribute Matching**:
   - Score candidate matches using multiple attributes:
     - Spatial proximity (weighted highest)
     - Temporal proximity
     - Size similarity (if available in AIS data)
     - Course alignment (if available)
   - Calculate a composite matching score

4. **Assignment Optimization**:
   - Formulate as a bipartite graph matching problem
   - Use Hungarian algorithm (Kuhn-Munkres) to find optimal assignment
   - Alternatively, use greedy assignment with priority queue

#### 1.3 Confidence Scoring

- Calculate match confidence based on:
  - Spatial distance (closer = higher confidence)
  - Temporal distance (closer in time = higher confidence)
  - Size similarity (if available)
  - Detection confidence from the model

- Apply threshold to filter low-confidence matches

#### 1.4 Output Generation

- Create CSV with matched pairs:
  - `image_name`: Satellite image identifier
  - `timestamp`: Detection time
  - `vessel_latitude`: From satellite detection
  - `vessel_longitude`: From satellite detection
  - `mmsi`: From matched AIS message
  - `match_confidence`: Calculated confidence score

### Implementation Considerations

- **Handling Multiple Matches**:
  - One-to-one matching using assignment algorithms
  - Resolve conflicts by prioritizing highest confidence matches

- **Handling Missing AIS Data**:
  - Flag detections without AIS matches as "unidentified vessels"
  - Consider these for potential illegal fishing or non-reporting vessels

- **Performance Optimization**:
  - Use spatial indexing (R-tree, KD-tree) for efficient proximity searches
  - Parallelize matching process for large datasets

## 2. Path Interpolation Methodology

### Overview
Path interpolation reconstructs continuous vessel tracks from discrete AIS points, providing a complete picture of vessel movements even when AIS reporting is sparse.

### Algorithm Steps

#### 2.1 Track Segmentation

- Group AIS messages by MMSI
- Sort messages chronologically
- Identify gaps in reporting (time intervals exceeding threshold)
- Split into separate track segments when gaps are too large

#### 2.2 Interpolation Methods

1. **Linear Interpolation**:
   - Simplest approach for short gaps
   - Interpolate latitude and longitude separately
   - Calculate intermediate positions at regular intervals

2. **Great-Circle Interpolation**:
   - More accurate for longer distances
   - Accounts for Earth's curvature
   - Uses spherical trigonometry to calculate points along great circle path

3. **Piecewise Cubic Hermite Interpolation**:
   - Preserves shape of trajectory
   - Maintains continuity in first derivatives
   - Better handles changes in speed and direction

4. **Kinematic Motion Models**:
   - Constant velocity model for straight paths
   - Constant turn rate model for curved paths
   - Incorporates speed and course information from AIS

#### 2.3 Speed and Course Interpolation

- **Speed Interpolation**:
  - Linear interpolation between known speed points
  - Account for acceleration/deceleration patterns
  - Consider vessel type for realistic speed changes

- **Course Interpolation**:
  - Use angular interpolation methods
  - Handle course wrapping (0° to 360°)
  - Ensure smooth turns based on vessel dynamics

#### 2.4 Adaptive Interpolation

- Select interpolation method based on:
  - Gap duration (longer gaps may need more sophisticated methods)
  - Vessel type (different vessels have different movement patterns)
  - Geographic context (coastal vs open ocean)
  - Speed and course changes before and after gap

#### 2.5 Output Generation

- Create CSV with interpolated track points:
  - `timestamp`: Time of interpolated position
  - `path_id`: Unique identifier for vessel track
  - `point_id`: Sequential identifier along path
  - `point_latitude`: Interpolated latitude
  - `point_longitude`: Interpolated longitude
  - `speed_on_ground`: Interpolated speed
  - `course_on_ground`: Interpolated course/heading
  - `interpolated`: Flag indicating if point is interpolated (1) or original (0)

### Implementation Considerations

- **Handling Special Cases**:
  - Coastline constraints (prevent land crossings)
  - Port entries and exits
  - Anchored vessels (minimal movement)

- **Uncertainty Estimation**:
  - Calculate confidence intervals for interpolated positions
  - Increase uncertainty with gap duration
  - Visualize uncertainty as error ellipses

- **Validation Methods**:
  - Cross-validation by removing known AIS points and comparing with interpolation
  - Calculate RMSE between interpolated and actual positions
  - Optimize interpolation parameters to minimize error

## 3. Integration with Existing Pipeline

### Data Flow

1. **Vessel Detection**:
   - Process Sentinel-1/2 imagery through existing detection pipeline
   - Generate vessel detections with coordinates, timestamp, and size

2. **AIS Correlation**:
   - Input: Detection results + AIS data
   - Process: Apply correlation algorithm
   - Output: Matched vessel identities (MMSI)

3. **Path Interpolation**:
   - Input: Correlated AIS points
   - Process: Apply interpolation algorithm
   - Output: Continuous vessel tracks

4. **Output Formatting**:
   - Convert detection results to GeoJSON and shapefile formats
   - Generate correlation and interpolation CSVs
   - Validate against required format specifications

### Implementation Strategy

1. **Modular Design**:
   - Create separate modules for correlation and interpolation
   - Define clear interfaces between components
   - Enable independent testing and validation

2. **Incremental Development**:
   - Implement basic correlation algorithm first
   - Add refinements and optimizations iteratively
   - Develop interpolation methods in parallel

3. **Performance Considerations**:
   - Optimize for large datasets (millions of AIS messages)
   - Consider distributed processing for production scale
   - Implement caching strategies for repeated queries

## 4. Evaluation Framework

### Correlation Evaluation

- **Metrics**:
  - Precision: Correctly matched vessels / Total matched vessels
  - Recall: Correctly matched vessels / Total actual matches possible
  - F1-score: Harmonic mean of precision and recall

- **Validation Method**:
  - Use ground truth dataset with known vessel identities
  - Calculate confusion matrix
  - Optimize parameters to maximize F1-score

### Interpolation Evaluation

- **Metrics**:
  - RMSE: Root Mean Square Error between interpolated and actual positions
  - Path deviation: Average distance between interpolated and actual paths
  - Temporal accuracy: Error in estimated arrival/departure times

- **Validation Method**:
  - Hold-out validation: Remove random AIS points and interpolate
  - Compare interpolated positions with actual positions
  - Analyze error patterns to improve algorithm