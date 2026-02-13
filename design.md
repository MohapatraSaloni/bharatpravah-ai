# BharatPravah - Design Document

## System Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          WEB UI LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Video Stream │  │   Metrics    │  │ Recommenda-  │  ┌────────┐ │
│  │   Display    │  │   Display    │  │ tion Display │  │ML Fore-│ │
│  │  (MJPEG)     │  │   (JSON)     │  │    (JSON)    │  │cast    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────┘ │
└────────────┬────────────────┬────────────────┬──────────────┬──────┘
             │                │                │              │
             │ HTTP/MJPEG     │ REST API       │ REST API     │ REST API
             │                │                │              │
┌────────────┴────────────────┴────────────────┴──────────────┴──────┐
│                     FASTAPI API LAYER                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  main.py - API Endpoints & Request Routing                   │  │
│  │  /stream/detect  /metrics  /recommendation  /ml/predict  /ui │  │
│  └────┬─────────────────┬──────────────────┬──────────────┬─────┘  │
└───────┼─────────────────┼──────────────────┼──────────────┼────────┘
        │                 │                  │              │
┌───────┴─────────────────┴──────────────────┴──────────────┴────────┐
│                    PROCESSING PIPELINE (per site)                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. VIDEO INGESTION                                           │  │
│  │    ┌──────────────────┐                                      │  │
│  │    │  VideoStream     │  ← MP4 files / RTSP streams          │  │
│  │    │  (Threading)     │    Resize: 960x540                   │  │
│  │    └────────┬─────────┘    Buffer: Latest frame              │  │
│  └─────────────┼──────────────────────────────────────────────────┘  │
│                ↓                                                     │
│  ┌─────────────┼──────────────────────────────────────────────────┐  │
│  │ 2. YOLO DETECTION                                            │  │
│  │    ┌────────┴─────────┐                                      │  │
│  │    │  YoloDetector    │  Model: YOLOv8n (nano)               │  │
│  │    │  (detector.py)   │  Classes: car, truck, bus,           │  │
│  │    └────────┬─────────┘  motorcycle, bicycle, person         │  │
│  │             │             Confidence: 0.35                    │  │
│  │             │             Process: Every 2nd frame            │  │
│  └─────────────┼──────────────────────────────────────────────────┘  │
│                ↓ [detections: bbox, class, confidence]              │
│  ┌─────────────┼──────────────────────────────────────────────────┐  │
│  │ 3. ROI MAPPING                                               │  │
│  │    ┌────────┴─────────┐                                      │  │
│  │    │ assign_detections│  Map detections to approaches        │  │
│  │    │  _to_rois()      │  (north, south, east, west)          │  │
│  │    └────────┬─────────┘  Using bbox center point             │  │
│  └─────────────┼──────────────────────────────────────────────────┘  │
│                ↓ [counts per ROI: vehicles, pedestrians]            │
│  ┌─────────────┼──────────────────────────────────────────────────┐  │
│  │ 4. METRICS ENGINE                                            │  │
│  │    ┌────────┴─────────┐                                      │  │
│  │    │ RollingAverager  │  Window: 10 frames                   │  │
│  │    │ compute_metrics  │  Density = vehicles / roi_area       │  │
│  │    └────────┬─────────┘  Score = min(1, density/threshold)   │  │
│  │             │             Update: Every 1 second              │  │
│  └─────────────┼──────────────────────────────────────────────────┘  │
│                ↓ [metrics: density, congestion_score]               │
│  ┌─────────────┼──────────────────────────────────────────────────┐  │
│  │ 5. RECOMMENDATION ENGINE                                     │  │
│  │    ┌────────┴─────────┐                                      │  │
│  │    │ build_recommenda-│  Cycle: 90s fixed                    │  │
│  │    │ tion() (advisor) │  Min Green: 20s per approach         │  │
│  │    └────────┬─────────┘  Smoothing: α=0.3                    │  │
│  │             │             Update: Every 10 seconds            │  │
│  └─────────────┼──────────────────────────────────────────────────┘  │
│                ↓ [recommendation: green times + reasoning]          │
│  ┌─────────────┼──────────────────────────────────────────────────┐  │
│  │ 6. LOGGER                                                    │  │
│  │    ┌────────┴─────────┐                                      │  │
│  │    │ Metrics History  │  Store: Last 60 seconds              │  │
│  │    │ Buffer (deque)   │  Format: Timestamped metrics         │  │
│  │    └────────┬─────────┘  Purpose: ML feature extraction      │  │
│  └─────────────┼──────────────────────────────────────────────────┘  │
│                ↓ [history: time-series metrics]                     │
│  ┌─────────────┼──────────────────────────────────────────────────┐  │
│  │ 7. ML FORECASTER                                             │  │
│  │    ┌────────┴─────────┐                                      │  │
│  │    │ OFFLINE TRAINING │  Input: Historical traffic logs      │  │
│  │    │ (train_forecaster│  Features: mean, std, trend          │  │
│  │    │  .py)            │  Model: Scikit-learn regression      │  │
│  │    └────────┬─────────┘  Output: forecaster.pkl              │  │
│  │             │                                                 │  │
│  │    ┌────────┴─────────┐                                      │  │
│  │    │ ONLINE PREDICT   │  Input: Last 60s metrics             │  │
│  │    │ (ml_layer.py)    │  Horizon: 30 seconds ahead           │  │
│  │    └──────────────────┘  Output: Predicted congestion        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### AWS Cloud Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS CLOUD DEPLOYMENT                        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  USERS (Traffic Control Room, Smart City Dashboard)         │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                     │
│                               ↓                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  API GATEWAY (Optional)                                    │    │
│  │  - Rate limiting, authentication                           │    │
│  │  - Custom domain, SSL/TLS                                  │    │
│  └────────────────────────┬───────────────────────────────────┘    │
│                           │                                         │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  ELASTIC LOAD BALANCER (ALB)                               │    │
│  │  - Distribute traffic across instances                     │    │
│  │  - Health checks, auto-scaling triggers                    │    │
│  └────────────┬───────────────────────┬───────────────────────┘    │
│               │                       │                             │
│               ↓                       ↓                             │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │  EC2 / ECS         │  │  EC2 / ECS         │  (Auto-scaling)    │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │                    │
│  │  │ FastAPI App  │  │  │  │ FastAPI App  │  │                    │
│  │  │ + YOLOv8     │  │  │  │ + YOLOv8     │  │                    │
│  │  │ + ML Model   │  │  │  │ + ML Model   │  │                    │
│  │  └──────────────┘  │  │  └──────────────┘  │                    │
│  └─────────┬──────────┘  └─────────┬──────────┘                    │
│            │                       │                                │
│            └───────────┬───────────┘                                │
│                        │                                            │
│         ┌──────────────┼──────────────┬─────────────┐              │
│         ↓              ↓              ↓             ↓              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐   │
│  │ S3 BUCKET  │ │ CLOUDWATCH │ │ SAGEMAKER  │ │ CLOUDWATCH   │   │
│  │            │ │            │ │ (Optional) │ │ LOGS         │   │
│  │ - Video    │ │ - Metrics  │ │            │ │              │   │
│  │   files    │ │ - Logs     │ │ - Model    │ │ - App logs   │   │
│  │ - ML model │ │ - Alarms   │ │   training │ │ - Error      │   │
│  │ - Config   │ │ - Dashbrd  │ │ - Endpoint │ │   tracking   │   │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘   │
│                                                                     │
│  DEPLOYMENT OPTIONS:                                                │
│  1. EC2: Direct instance deployment (t3.medium or GPU instances)   │
│  2. ECS: Containerized deployment with Docker                      │
│  3. ECS Fargate: Serverless container deployment                   │
│                                                                     │
│  SCALING STRATEGY:                                                  │
│  - Horizontal: Auto-scaling group based on CPU/memory              │
│  - Vertical: GPU instances (g4dn.xlarge) for faster inference     │
│  - Multi-region: Deploy in multiple AWS regions for HA            │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Choices

### 1. Fixed 90-Second Cycle Time

**Rationale:** Standard traffic signal cycle time used in Indian cities. Provides predictability for drivers and pedestrians while allowing sufficient time for all approaches.

**Benefits:**
- Compatible with existing traffic infrastructure
- Familiar to traffic engineers and operators
- Balances responsiveness with stability
- Simplifies timing calculations

### 2. Minimum Green Time (20 seconds)

**Rationale:** Safety constraint to ensure sufficient crossing time for vehicles and pedestrians.

**Benefits:**
- Prevents unsafe short green phases
- Ensures pedestrian safety (typical crossing time: 15-20s)
- Meets traffic engineering standards
- Reduces driver frustration and violations

### 3. Temporal Smoothing (α = 0.3)

**Rationale:** Prevents abrupt changes in signal timing that could confuse drivers or cause safety issues.

**Algorithm:**
```python
new_green = 0.7 * previous_green + 0.3 * calculated_green
```

**Benefits:**
- Gradual transitions between timing plans
- Reduces driver confusion
- Filters out temporary fluctuations
- Maintains system stability

### 4. Update Every 10 Seconds

**Rationale:** Balances responsiveness to traffic changes with computational efficiency and stability.

**Benefits:**
- Frequent enough to adapt to changing conditions
- Infrequent enough to avoid excessive computation
- Allows 9 updates per 90-second cycle
- Provides smooth adaptation without flickering

### 5. Explainability First

**Rationale:** Traffic engineers and judges need to understand and trust the system's decisions.

**Implementation:**
- Human-readable reasoning for each recommendation
- Transparent metric calculations
- Clear demand-based allocation logic
- Configurable parameters with documentation

**Example Output:**
```json
{
  "reason": "More green given to 'north' due to higher demand (vehicles + pedestrian factor).",
  "demand": {"north": 18.5, "south": 7.0}
}
```

### 6. Modular Multi-Site Architecture

**Rationale:** Scalability and maintainability for city-wide deployment.

**Design:**
- Independent VisionRuntime per site
- No cross-site dependencies
- Site-specific configuration (ROIs, video sources)
- Parallel processing capability

**Benefits:**
- Easy to add new intersections
- Fault isolation (one site failure doesn't affect others)
- Horizontal scaling (distribute sites across servers)
- Flexible deployment (cloud or on-premise)


## API Endpoints

### 1. GET /

**Purpose:** System health check and status

**Response:**
```json
{
  "message": "BharatPravah backend running",
  "sites": ["blr", "delhi", "kolkata"],
  "ml_model_loaded": true,
  "ml_model_error": null
}
```

**Use Case:** Monitoring, health checks, service discovery

---

### 2. GET /detections?site_id={site_id}

**Purpose:** Get latest detection results for a site

**Parameters:**
- `site_id` (required): Site identifier (e.g., "blr")

**Response:**
```json
{
  "site_id": "blr",
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.87,
      "bbox": [120.5, 200.3, 180.2, 250.8]
    }
  ]
}
```

**Use Case:** Debugging, validation, external integrations

---

### 3. GET /metrics?site_id={site_id}

**Purpose:** Get current congestion metrics

**Parameters:**
- `site_id` (required): Site identifier

**Response:**
```json
{
  "ts": 1234567890.123,
  "site_id": "blr",
  "status": "ok",
  "counts": {
    "north": {"vehicles": 15, "pedestrians": 3},
    "south": {"vehicles": 8, "pedestrians": 1}
  },
  "metrics": {
    "north": {
      "vehicles": 15,
      "pedestrians": 3,
      "density": 0.00008,
      "congestion_score": 0.67,
      "roi_area": 187500
    },
    "south": {
      "vehicles": 8,
      "pedestrians": 1,
      "density": 0.00004,
      "congestion_score": 0.33,
      "roi_area": 200000
    }
  },
  "density_threshold": 0.00012,
  "rois": {
    "north": [8, 160, 836, 333],
    "south": [7, 338, 1156, 576]
  }
}
```

**Update Frequency:** Every 1 second

**Use Case:** Dashboard display, analytics, monitoring

---

### 4. GET /recommendation?site_id={site_id}

**Purpose:** Get adaptive signal timing recommendation

**Parameters:**
- `site_id` (required): Site identifier

**Response:**
```json
{
  "ts": 1234567890.123,
  "site_id": "blr",
  "status": "ok",
  "cycle_s": 90,
  "min_green_s": 20,
  "greens": {
    "north": 55,
    "south": 35
  },
  "demand": {
    "north": 18.5,
    "south": 7.0
  },
  "reason": "More green given to 'north' due to higher demand (vehicles + pedestrian factor).",
  "updated_every_s": 10
}
```

**Update Frequency:** Every 10 seconds

**Use Case:** Signal controller integration, operator dashboard

---

### 5. GET /ml/predict?site_id={site_id}

**Purpose:** Get ML-based congestion forecast

**Parameters:**
- `site_id` (required): Site identifier

**Response:**
```json
{
  "status": "ok",
  "ts": 1234567890.123,
  "site_id": "blr",
  "horizon_s": 30,
  "forecast": {
    "north": 0.72,
    "south": 0.38
  },
  "model": "RandomForestRegressor"
}
```

**Requirements:** Minimum 60 seconds of metrics history

**Use Case:** Predictive analytics, proactive signal adjustment

---

### 6. GET /stream/detect?site_id={site_id}

**Purpose:** Live MJPEG video stream with detection overlays

**Parameters:**
- `site_id` (required): Site identifier

**Response:** Multipart MJPEG stream (Content-Type: multipart/x-mixed-replace)

**Frame Rate:** ~15-20 FPS (depends on processing speed)

**Use Case:** Live monitoring, operator dashboard, demo

---

### 7. GET /ui

**Purpose:** Web-based dashboard interface

**Response:** HTML page with embedded JavaScript

**Features:**
- Site selection dropdown
- Live video stream
- Real-time metrics display
- Signal timing recommendations
- ML forecast visualization
- Auto-refresh (1-2 second intervals)

**Use Case:** Traffic control room, demonstrations, monitoring


## Data Flow Diagrams

### Complete System Data Flow

```
┌─────────────┐
│ Video Input │ (MP4 / RTSP)
└──────┬──────┘
       │
       ↓ [Raw frames: 1920x1080 @ 25fps]
┌──────────────────┐
│ Video Ingestion  │ Resize to 960x540, Buffer latest frame
└──────┬───────────┘
       │
       ↓ [Resized frame: 960x540]
┌──────────────────┐
│ YOLO Detection   │ Process every 2nd frame, Confidence > 0.35
└──────┬───────────┘
       │
       ↓ [Detections: [{class, bbox, conf}, ...]]
┌──────────────────┐
│ ROI Mapping      │ Assign detections to approaches using bbox center
└──────┬───────────┘
       │
       ↓ [Counts: {north: {vehicles: 15, peds: 3}, south: {...}}]
┌──────────────────┐
│ Rolling Average  │ Smooth counts over 10-frame window
└──────┬───────────┘
       │
       ↓ [Smoothed counts]
┌──────────────────┐
│ Metrics Engine   │ Calculate density & congestion score
└──────┬───────────┘
       │
       ├─────────────────────────────────────┐
       │                                     │
       ↓ [Metrics payload]                  ↓ [Store in history buffer]
┌──────────────────┐                  ┌──────────────────┐
│ Recommendation   │                  │ Metrics Logger   │
│ Engine           │                  │ (60s buffer)     │
└──────┬───────────┘                  └──────┬───────────┘
       │                                     │
       ↓ [Green times + reasoning]          ↓ [Time-series metrics]
┌──────────────────┐                  ┌──────────────────┐
│ API Response     │                  │ ML Forecaster    │
│ /recommendation  │                  │ Feature Extract  │
└──────────────────┘                  └──────┬───────────┘
                                             │
                                             ↓ [Feature vector]
                                      ┌──────────────────┐
                                      │ ML Model Predict │
                                      │ (forecaster.pkl) │
                                      └──────┬───────────┘
                                             │
                                             ↓ [Forecast: 30s ahead]
                                      ┌──────────────────┐
                                      │ API Response     │
                                      │ /ml/predict      │
                                      └──────────────────┘
```

### Metrics Calculation Flow

```
Detections → ROI Assignment → Count Aggregation → Rolling Average → 
Density Calculation → Congestion Score → API Response

Details:
1. ROI Assignment: Check if bbox center (cx, cy) falls within ROI rectangle
2. Count Aggregation: Separate vehicles and pedestrians per ROI
3. Rolling Average: Maintain 10-frame deque per ROI, compute mean
4. Density: vehicles / roi_area_pixels
5. Congestion Score: min(1.0, density / threshold)
```

### Recommendation Generation Flow

```
Metrics → Demand Calculation → Proportional Split → Constraint Enforcement → 
Smoothing → Normalization → Rounding → API Response

Details:
1. Demand = vehicles + 0.5 * pedestrians
2. Proportional Split: green_time = (demand / total_demand) * 90s
3. Constraint Enforcement: clamp(green_time, 20s, 70s)
4. Smoothing: 0.7 * prev_green + 0.3 * new_green
5. Normalization: Ensure sum(greens) = 90s
6. Rounding: Round to integer, adjust max-demand approach for exact sum
```

### ML Forecast Flow

```
Metrics History (60s) → Feature Extraction → Model Prediction → 
Clipping [0,1] → API Response

Features per approach:
- mean_congestion_last_60s
- std_congestion_last_60s
- trend_congestion (linear regression slope)
- current_congestion
- mean_vehicle_count
- mean_pedestrian_count

Total features: 6 * num_approaches (e.g., 12 for 2 approaches)
```


## Component Design Details

### 1. VisionRuntime (`vision_runtime.py`)

**Purpose:** Orchestrates the entire processing pipeline for a single site

**Key Responsibilities:**
- Manage video stream thread
- Coordinate detection, metrics, and recommendation generation
- Maintain frame buffer and detection results
- Store metrics history for ML forecasting
- Thread-safe access to latest results

**Design Pattern:** Producer-Consumer with thread synchronization

**Threading Model:**
- Main thread: Video capture and detection
- Timed callbacks: Metrics computation (1s), Recommendations (10s)
- Lock-based synchronization for shared state

### 2. YoloDetector (`detector.py`)

**Purpose:** Vehicle and pedestrian detection using YOLOv8

**Algorithm:**
- Model: YOLOv8n (nano - fastest variant)
- Input: BGR frame from OpenCV
- Output: List of detections with class, confidence, bbox

**Filtering:**
- Keep only traffic-relevant classes: person, bicycle, car, motorcycle, bus, truck
- Confidence threshold: 0.35 (configurable)

**Performance Optimization:**
- Process every Nth frame (default: 2)
- Use nano model for speed
- Disable verbose logging

### 3. Metrics Computer (`metrics.py`)

**Purpose:** Convert detections into actionable congestion metrics

**Processing Pipeline:**
```
Detections → ROI Assignment → Count Aggregation → 
Rolling Average → Density Calculation → Congestion Score
```

**Key Algorithms:**

**ROI Assignment:**
- Use bounding box center point
- Check if center falls within ROI rectangle
- Assign to first matching ROI (no overlap handling)

**Congestion Score Calculation:**
```python
density = vehicle_count / roi_area_pixels
congestion_score = min(1.0, density / DENSITY_THRESHOLD)
```

**Rolling Average:**
- Window size: 10 frames
- Separate buffers for vehicles and pedestrians
- Prevents metric flickering

### 4. Signal Timing Advisor (`advisor.py`)

**Purpose:** Generate adaptive signal timing recommendations

**Algorithm Steps:**

**Step 1: Demand Calculation**
```python
demand = vehicles + (ped_weight * pedestrians)
# ped_weight = 0.5 (configurable)
```

**Step 2: Proportional Allocation**
```python
share = demand[approach] / total_demand
raw_green = share * cycle_time
```

**Step 3: Constraint Enforcement**
```python
clamped_green = clamp(raw_green, min_green, max_green)
# min_green = 20s, max_green = 70s (for 90s cycle)
```

**Step 4: Normalization**
```python
# Ensure sum(greens) = cycle_time
normalized_green = (clamped_green / sum_clamped) * cycle_time
```

**Step 5: Temporal Smoothing**
```python
smoothed = (1 - alpha) * prev_green + alpha * new_green
# alpha = 0.3 (30% new, 70% previous)
```

**Step 6: Rounding & Adjustment**
- Round to integer seconds
- Adjust highest-demand approach to fix rounding errors

### 5. ML Forecaster (`ml_layer.py`, `ml_features.py`)

**Purpose:** Predict congestion 30 seconds ahead

**Model:** Scikit-learn regression model (trained separately)

**Feature Engineering:**
```python
# For each approach (north, south):
features = [
  mean_congestion_last_60s,
  std_congestion_last_60s,
  trend_congestion,  # linear regression slope
  current_congestion,
  mean_vehicle_count,
  mean_pedestrian_count
]
# Total features: 6 * num_approaches
```

**Prediction Process:**
1. Collect last 60 seconds of metrics history
2. Extract features per approach
3. Concatenate into feature vector
4. Feed to trained model
5. Clip predictions to [0, 1] range

**Graceful Degradation:**
- Return "model_not_loaded" if model file missing
- Return "insufficient_history" if < 60s of data
- Never crash the main pipeline

### 6. Video Stream (`video_stream.py`)

**Purpose:** Efficient video frame capture with threading

**Design:**
- Separate thread for frame reading
- Circular buffer (size 1) for latest frame
- Non-blocking read for main processing thread

**Benefits:**
- Prevents frame read blocking detection
- Maintains real-time performance
- Handles video loop for demo footage

### 7. Site Configuration (`site_config.py`)

**Purpose:** Centralized multi-site configuration

**Structure:**
```python
SITES = {
  "blr": {
    "name": "Bangalore - MG Road Junction",
    "video_path": "data/blr.mp4",
    "rois": {
      "north": (8, 160, 836, 333),
      "south": (7, 338, 1156, 576)
    }
  }
}
```

**Extensibility:**
- Add new sites by adding dictionary entries
- No code changes required
- ROI coordinates site-specific

