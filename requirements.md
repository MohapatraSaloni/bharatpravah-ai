# BharatPravah - Requirements Document

## Project Overview
**Project Name:** BharatPravah  
**Type:** AI-based Adaptive Traffic Management Prototype  
**Purpose:** Software-only solution for intelligent traffic signal control using existing CCTV infrastructure

## Functional Requirements

### FR1: Computer Vision Detection
- Detect vehicles and pedestrians from CCTV footage in real-time
- Support multiple vehicle classes: car, truck, bus, motorcycle, bicycle
- Detect pedestrians separately for safety prioritization
- Minimum confidence threshold: 35%
- Process every 2nd frame for performance optimization

### FR2: Multi-Site Support
- Support multiple traffic intersections simultaneously
- Independent processing pipeline per site
- Site-specific ROI (Region of Interest) configuration
- Configurable video sources per site

### FR3: Congestion Metrics
- Calculate real-time congestion metrics per approach (north, south, etc.)
- Metrics include:
  - Vehicle count
  - Pedestrian count
  - Density (vehicles per pixel²)
  - Normalized congestion score (0-1 scale)
- Update metrics every 1 second
- Apply rolling average smoothing (10-frame window)

### FR4: Adaptive Signal Timing
- Generate signal timing recommendations based on real-time congestion
- Fixed cycle time: 90 seconds
- Update recommendations every 10 seconds
- Constraints:
  - Minimum green time: 20 seconds per approach
  - Sum of all green times must equal cycle time
- Apply smoothing to prevent abrupt timing changes
- Provide explainable reasoning for recommendations

### FR5: ML-Based Forecasting
- Predict congestion 30 seconds ahead
- Use historical metrics (last 60 seconds)
- Forecast for all configured approaches
- Graceful degradation if model unavailable

### FR6: REST API Endpoints
- `GET /` - System status and site list
- `GET /detections?site_id=X` - Latest detection results
- `GET /metrics?site_id=X` - Current congestion metrics
- `GET /recommendation?site_id=X` - Signal timing recommendation
- `GET /ml/predict?site_id=X` - ML congestion forecast
- `GET /stream/detect?site_id=X` - Live MJPEG video stream with bounding boxes
- `GET /ui` - Web dashboard interface

### FR7: Web Dashboard
- Site selection dropdown
- Live video feed with detection overlays
- Real-time metrics display
- Current signal timing recommendations
- ML forecast visualization
- Auto-refresh every 1-2 seconds
- Responsive design

## Non-Functional Requirements

### NFR1: Performance
- Process video at minimum 15 FPS (after frame skipping)
- API response time < 100ms for metrics/recommendations
- Support at least 3 concurrent sites on standard hardware

### NFR2: Scalability
- Cloud-ready architecture
- Stateless API design
- Independent site processing (horizontal scaling potential)

### NFR3: Explainability
- All recommendations must include human-readable reasoning
- Transparent metric calculations
- Configurable parameters with clear documentation

### NFR4: Reliability
- Graceful handling of missing video frames
- Fallback behavior when ML model unavailable
- Warm-up period handling for initial metrics

### NFR5: Maintainability
- Modular architecture (separate detector, metrics, advisor, ML components)
- Configuration-driven site setup
- Clear separation of concerns

## Technical Constraints

### TC1: Hardware Constraints
- Software-only solution (no new hardware required)
- Must work with existing CCTV infrastructure
- Use recorded video for prototype demonstration

### TC2: Technology Stack
- Backend: Python 3.10+, FastAPI
- Computer Vision: YOLOv8n (lightweight model)
- ML Framework: scikit-learn
- Video Processing: OpenCV

### TC3: Model Constraints
- Use pre-trained YOLOv8n for detection (no custom training required)
- ML forecaster trained on historical traffic data
- Model file: `models/forecaster.pkl`

## Data Requirements

### DR1: Input Data
- Video format: MP4 or live RTSP stream
- Resolution: Flexible (resized to 960x540 for processing)
- Frame rate: 25-30 FPS

### DR2: Configuration Data
- Site configuration: video path, ROI coordinates, site name
- ROI format: (x1, y1, x2, y2) pixel coordinates
- Density threshold for congestion scoring

### DR3: Output Data
- Detection results: class, confidence, bounding box
- Metrics: JSON format with timestamp
- Recommendations: JSON with green times and reasoning
- Forecasts: JSON with predicted congestion scores

## User Requirements

### UR1: Target Users
- Traffic control room operators
- Smart city administrators
- Traffic engineers
- System integrators

### UR2: Usability
- Simple web interface requiring no training
- Clear visual indicators (Low/Moderate/High congestion)
- Real-time updates without manual refresh
- Multi-site switching without page reload

## Compliance & Safety

### CS1: Safety Requirements
- Minimum green time enforcement (20s) for safety
- Pedestrian consideration in timing calculations
- Smooth transitions to prevent sudden changes

### CS2: Operational Requirements
- System must indicate warm-up status
- Clear error messages for missing data
- Status indicators for ML model availability

## Success Criteria

1. Accurate vehicle/pedestrian detection (>80% precision)
2. Stable congestion metrics (minimal flickering)
3. Reasonable signal timing recommendations (validated by traffic engineers)
4. ML forecast correlation with actual congestion (>70%)
5. System uptime >99% during demonstration
6. Dashboard loads and updates smoothly
7. Support 3+ sites simultaneously without performance degradation

## Future Enhancements (Out of Scope for Prototype)

- Integration with actual traffic signal controllers
- Historical data analytics and reporting
- Traffic incident detection
- Emergency vehicle prioritization
- Mobile application
- Advanced ML models (deep learning forecasting)
- Multi-intersection coordination
- Weather condition integration
