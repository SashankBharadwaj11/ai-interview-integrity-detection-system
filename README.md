🎥🔍 AI Interview Integrity Detection System
Offline Audio + Video Analysis for Exam/Interview Monitoring

This project is an AI-driven integrity analysis system that processes offline video + audio recordings to detect suspicious behavior during online interviews or exam sessions.

It generates a detailed Session Integrity Report with:

🧑‍💻 Face presence tracking

👀 Eye & gaze direction analysis

🧭 Blink/EAR tracking

🧍‍♂️ Multi-face detection

📱📚 Forbidden object detection

🔊 Speaker-consistency scoring using WavLM

📊 Timeline visualizations (Chart.js)

⭐ Overall integrity score (0–100)

Everything runs locally on the user’s machine — no data leaves the system.

🚀 Key Features
🎞️ Video Analysis

Face detection (MTCNN, facenet-pytorch)

Eye tracking (MediaPipe FaceMesh)

Gaze classification: left, right, center, up, down

EAR-based blink detection

Face-missing event alerts

Multi-face detection (detect extra persons)

FPS-aware optimizations

📦 Object Detection

Model: YOLOv8-Nano (Ultralytics)
Detects:

📱 Mobile phones

📚 Books/notes

📝 Paper sheets

(Easily extendable via object_detection.py)

🔊 Audio Integrity Analysis

Powered by WavLM-Base+ (HuggingFace).

Pipeline:

Extract audio (ffmpeg)

Split into chunks (2–3 seconds)

Generate embeddings per chunk

Compute:

Average similarity

Minimum chunk similarity

Speaker change probability

Final Speaker Consistency Score (0–100)

📑 Session Integrity Report

Generated via report_generator.py.

Includes:

Alerts summary

Object detection hits

Video activity timeline

Speaker consistency graph

Weighted combined integrity score

Auto-saved JSON at logs/sessions/

Rendered with Flask templates

🧱 Project Architecture
ai-interview-integrity-detection-system/
│
├── src/
│   ├── dashboard/                  # Flask Web UI
│   │   ├── app.py
│   │   └── templates/
│   │       ├── dashboard.html
│   │       ├── upload.html
│   │       └── session_report.html
│   │
│   ├── detection/                  # Video detection modules
│   │   ├── face_detection.py
│   │   ├── eye_tracking.py
│   │   ├── object_detection.py
│   │   └── multi_face.py
│   │
│   ├── audio/                      # Audio pipeline
│   │   ├── speaker_consistency.py
│   │   └── utils_audio.py
│   │
│   ├── analysis/                   # Scoring logic
│   │   ├── scoring.py
│   │   └── report_generator.py
│   │
│   ├── utils/
│   │   ├── logging.py
│   │   ├── screenshot_utils.py
│   │   └── timer.py
│   │
│   ├── offline_processor.py        # Main offline pipeline
│   └── config.yaml                 # Detection parameters
│
├── uploads/                         # User-uploaded video/audio
├── logs/
│   └── sessions/                    # JSON reports
│
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Create Conda Environment
conda create -n interview310 python=3.10
conda activate interview310

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install ffmpeg (Required)

macOS:

brew install ffmpeg

🧪 Running the Application

Start the Flask dashboard:

python -m src.dashboard.app


Then visit:

http://127.0.0.1:5000

📤 How to Use the System

Upload pre-recorded video

Upload associated audio file (recommended same duration)

Click Analyze Recording

Processing takes about 1–2 minutes per 15-minute video

View the full Session Integrity Report

📊 Scoring System
🎞️ Video Integrity Score (0–100)

Penalizes:

Face missing

Gaze away (L/R/U/D)

Excessive eye movement

Multiple faces

Forbidden objects

🔊 Audio Integrity Score (0–100)

Based on WavLM similarity:

✔ High similarity → same speaker

❌ Sudden drops → possible speaker change

❗ speaker_change_flag = True → penalty applied

⭐ Overall Score
overall_score = 0.7 * video_score + 0.3 * audio_score

🧠 Models Used
Task	Model	Framework
Face Detection	MTCNN	facenet-pytorch
Eye/Gaze Tracking	FaceMesh	MediaPipe
Object Detection	YOLOv8n	Ultralytics
Speaker Embeddings	WavLM-Base+	HuggingFace
Audio Extraction	ffmpeg	subprocess
🧩 Configuration

Modify detection behavior via:

src/config.yaml


Example:

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

🧼 Code Quality Enhancements

Unified scoring pipeline

Improved JSON schemas

Robust Jinja templates

Optimized YOLO inference

Modularized audio engine

Cleaner folder structure

Single unified conda environment

Fixed transformers & tokenizers conflicts

🛡️ Privacy Guarantee

✔ No cloud upload
✔ No logging of raw video/audio
✔ 100% offline processing
✔ Suitable for exams, interviews, assessments

📬 Future Enhancements

Real-time detection (live webcam)

OCR for reading notes on desk

Emotion recognition

Speaker diarization improvements

Deepfake voice detection

GPU-accelerated cloud API (FastAPI)
