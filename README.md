# VLM Surveillance Detection System

## Overview
This is a Real-Time Surveillance Alert System with VLM (Vision Language Model) Behavior Analysis. It detects suspicious objects (gun, knife, blunt_object) using YOLOv8 and analyzes behavior using Google Gemini 2.5 Flash.

## Project Structure

```
vlm/
├── surveillance_detection.py   # Main entry point - runs YOLO detection on webcam
├── vlm_behavior_analyzer.py    # Buffers frames, captures clips, sends to VLM API
├── vlm_api_server.py           # Flask API server for Gemini 2.5 Flash analysis
├── alert_system.py             # Email/desktop notification system
├── config.json                 # Configuration settings
├── best (epoch_100).pt         # YOLOv8 trained model
├── main.py                     # Test script for Gemma 3 model
├── main_2.py                   # Test script for Gemini API
├── alerts/                     # Saved alert images
├── vlm_analysis_clips/         # Saved video clips of detections
├── alerts.log                  # Detection history
└── alerts_db.json              # Alert database
```

## How It Works

### Architecture
```
Webcam → surveillance_detection.py (YOLOv8) → 
    ├── Detection → alert_system.py (Email Notifications)
    └── Suspicious Object → vlm_behavior_analyzer.py → vlm_api_server.py (Gemini 2.5 Flash)
```

### Workflow
1. **surveillance_detection.py** runs YOLOv8 on webcam feed in real-time
2. When suspicious object (gun, knife, blunt_object) detected with >80% confidence:
   - Saves alert image
   - Triggers VLM behavior analysis
3. **vlm_behavior_analyzer.py** captures 5-second clip at 3 FPS (15 frames)
4. Sends frames to **vlm_api_server.py** for Gemini analysis
5. **vlm_api_server.py** analyzes frames:
   - Stage 1: Quick yes/no check for suspicious activity
   - Stage 2: Detailed behavior analysis if suspicious
6. **alert_system.py** sends email/desktop notifications with VLM analysis

## How to Run

### Step 1: Start the VLM API Server
```bash
cd d:/miner_project_all_material/vlm
python vlm_api_server.py
```
Server runs on http://localhost:8000

### Step 2: Run the Surveillance Detection (in another terminal)
```bash
cd d:/miner_project_all_material/vlm
python surveillance_detection.py
```

### Optional: Custom Parameters
```bash
python surveillance_detection.py --model "path/to/model.pt" --camera 0
```

## Configuration (config.json)

| Setting | Value | Description |
|---------|-------|-------------|
| confidence_threshold | 0.8 | Min confidence to trigger alerts |
| alert_cooldown | 10 | Seconds between alerts |
| capture_duration | 5 | VLM clip length in seconds |
| capture_fps | 3 | VLM capture frame rate |
| capture_cooldown | 30 | Seconds between VLM analyses |
| suspicious_classes | gun, knife, blunt_object | Objects to monitor |

## Current Status (from alerts.log)

- ✅ System is detecting guns and blunt objects
- ✅ Email alerts configured but have MIME subtype errors
- ✅ Desktop notifications and sound alerts working
- ✅ VLM analysis triggered successfully

## Files Generated During Runtime

- `alerts/alert_YYYYMMDD_HHMMSS.jpg` - Alert images with bounding boxes
- `vlm_analysis_clips/clip_<object>_<timestamp>.mp4` - Video clips sent for VLM analysis
- `alerts.log` - Text log of all detections and alerts
- `alerts_db.json` - JSON database of all alerts

## Requirements

```
pip install -r req.txt
```

Required packages:
- pillow
- torch
- torchvision
- transformers
- opencv-python
- flask
- requests
- numpy

