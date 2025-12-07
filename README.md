# 🎥🔍 AI Interview Integrity Detection System  
*Offline Audio + Video Monitoring for Exams & Interviews*

This project performs **offline integrity analysis** on pre-recorded **video + audio** to identify suspicious behavior during exams or interviews.

It generates a detailed **Session Integrity Report** including:
- Face presence tracking  
- Eye & gaze direction  
- Blink rate / EAR  
- Object detection  
- Multi-face presence  
- Speaker consistency scoring  
- Timeline graphs  
- Combined integrity score  

All processing runs **locally** — no uploads or data sharing.

---

# 🚀 Features

## 🎞️ Video Analysis
- Face detection (MTCNN – facenet-pytorch)  
- Eye tracking & facial landmarks (MediaPipe FaceMesh)  
- Gaze direction classification (left/right/up/down/center)  
- Blink detection via EAR  
- Face-missing alerts  
- Multi-face detection  

## 📦 Object Detection
Model: **YOLOv8-Nano (Ultralytics)**  
Detects:
- 📱 Mobile phones  
- 📚 Books / notes  
- 📝 Papers  
- (Extendable via `object_detection.py`)  

## 🔊 Audio Integrity Analysis
Powered by **WavLM-Base+ (HuggingFace)**

Pipeline:
1. Extract audio (ffmpeg)  
2. Split audio (2–3 sec chunks)  
3. Compute embeddings  
4. Compare cosine similarity  
5. Detect speaker change  

Outputs:
- Average similarity  
- Minimum similarity  
- Speaker change flag  
- Audio integrity score (0–100)  

## 📑 Session Integrity Report
- Alerts summary  
- Object detection activity  
- Timeline graphs (Chart.js)  
- Speaker consistency graph  
- Combined score  
- Auto-saved JSON reports  
- Displayed via Flask dashboard  

---

# 🧱 Project Architecture

ai-interview-integrity-detection-system/
│
├── src/
│ ├── dashboard/ # Flask dashboard UI
│ │ ├── app.py
│ │ └── templates/ # HTML templates
│ │ ├── dashboard.html
│ │ ├── upload.html
│ │ └── session_report.html
│ │
│ ├── detection/ # Video detection modules
│ │ ├── face_detection.py
│ │ ├── eye_tracking.py
│ │ ├── object_detection.py
│ │ └── multi_face.py
│ │
│ ├── audio/ # Audio processing modules
│ │ ├── speaker_consistency.py
│ │ └── utils_audio.py
│ │
│ ├── analysis/ # Scoring + reporting logic
│ │ ├── scoring.py
│ │ └── report_generator.py
│ │
│ ├── utils/ # Utility helpers
│ │ ├── logging.py
│ │ ├── screenshot_utils.py
│ │ └── timer.py
│ │
│ ├── offline_processor.py # Full offline pipeline
│ └── config.yaml # Detection configuration
│
├── uploads/ # Uploaded video/audio
├── logs/
│ └── sessions/ # JSON session reports
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

