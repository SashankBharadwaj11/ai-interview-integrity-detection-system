# 🎥 AI Interview Integrity Detection System  
### Offline Audio + Video Monitoring for Exams & Interviews

This system performs **offline integrity analysis** on pre-recorded **video + audio** to detect suspicious behavior in online interviews or exam sessions.  
It generates a detailed **Session Integrity Report** containing:

- Face presence timeline  
- Eye & gaze direction  
- Blink/EAR activity  
- Forbidden object detection  
- Multi-face detection  
- Speaker consistency (WavLM)  
- Timeline charts (Chart.js)  
- Combined integrity score  

Everything is processed **locally** — no cloud upload or external servers.

---

# 📌 Table of Contents

1. [Features](#-features)  
2. [Project Architecture](#-project-architecture)  
3. [Installation](#-installation)  
4. [Running the Application](#-running-the-application)  
5. [How to Use](#-how-to-use)  
6. [Scoring System](#-scoring-system)  
7. [Models Used](#-models-used)  
8. [Configuration](#-configuration)  
9. [Privacy](#-privacy)  
10. [Future Enhancements](#-future-enhancements)

---

# 🚀 Features

## 🎞️ Video Analysis  
- Face detection (MTCNN)  
- Eye tracking (MediaPipe FaceMesh)  
- Gaze direction classification  
- Blink detection (EAR)  
- Face-missing alerts  
- Multi-face detection  

## 📦 Object Detection  
- YOLOv8-Nano  
- Detects: mobile phones, books, paper  
- FPS-aware optimized inference  

## 🔊 Audio Integrity Analysis  
- WavLM-Base+ embeddings  
- Chunk-based speaker similarity  
- Minimum/average similarity  
- Speaker change detection  
- Audio integrity score (0–100)  

## 📑 Session Integrity Report  
- Timeline graphs (Chart.js)  
- Speaker consistency graph  
- Gaze + object alerts  
- Combined score  
- JSON generated under `logs/sessions/`  
- Flask dashboard UI  

---

# 🧱 Project Architecture

```plaintext
ai-interview-integrity-detection-system/
│
├── src/
│   ├── dashboard/
│   │   ├── app.py
│   │   └── templates/
│   │       ├── dashboard.html
│   │       ├── upload.html
│   │       └── session_report.html
│   │
│   ├── detection/
│   │   ├── face_detection.py
│   │   ├── eye_tracking.py
│   │   ├── object_detection.py
│   │   └── multi_face.py
│   │
│   ├── audio/
│   │   ├── speaker_consistency.py
│   │   └── utils_audio.py
│   │
│   ├── analysis/
│   │   ├── scoring.py
│   │   └── report_generator.py
│   │
│   ├── utils/
│   │   ├── logging.py
│   │   ├── screenshot_utils.py
│   │   └── timer.py
│   │
│   ├── offline_processor.py
│   └── config.yaml
│
├── uploads/
├── logs/
│   └── sessions/
│
├── requirements.txt
└── README.md
---

# ⚙️ Installation

## 1️⃣ Create Conda Environment
```bash
conda create -n interview310 python=3.10
conda activate interview310
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Install ffmpeg (Required for audio extraction)
macOS:
bash
Copy code
brew install ffmpeg
🧪 Running the Application
Start the Flask dashboard:

bash
Copy code
python -m src.dashboard.app
Now open:

cpp
Copy code
http://127.0.0.1:5000
📤 How to Use the System
Upload video recording

Upload audio recording

Click Analyze Recording

Processing time: 1–2 min per 15 min video

View Session Integrity Report

📊 Scoring System
🎞️ Video Integrity Score (0–100)
Penalties for:

Face missing

Looking away (L/R/U/D)

Excessive eye movement

Multi-face detection

Forbidden objects

🔊 Audio Integrity Score (0–100)
Based on WavLM similarity:

High similarity = same speaker

Low similarity = possible switch

speaker_change_flag = True → penalty applied

⭐ Combined Overall Score
Formula:

python
Copy code
overall_score = 0.7 * video_score + 0.3 * audio_score
🧠 Models Used
Task	Model	Framework
Face Detection	MTCNN	facenet-pytorch
Eye Tracking	FaceMesh	MediaPipe
Object Detection	YOLOv8n	Ultralytics
Speaker Embeddings	WavLM-Base+	HuggingFace Transformers
Audio Extraction	ffmpeg	subprocess

🧩 Configuration (config.yaml)
Below is a sample config:

yaml
Copy code
detection:
  face:
    detection_interval: 5
    min_confidence: 0.8

  eyes:
    gaze_threshold: 2
    blink_threshold: 0.3

  objects:
    min_confidence: 0.65
    max_fps: 5

audio_monitoring:
  sample_rate: 16000
Modify these to customize system behavior.

🧼 Code Quality Improvements
Unified scoring pipeline

Robust JSON schema

Cleaner Jinja templates

YOLO inference optimization

Isolated audio subsystem

Environment fixes

Removed duplicate envs + conflicts

Support for command-line offline processing


📬 Future Enhancements
Real-time webcam detection

Emotion detection

OCR for desk notes

Multi-speaker diarization

Deepfake voice detection

GPU FastAPI deployment

