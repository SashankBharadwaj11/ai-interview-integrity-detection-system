🎥🔍 AI-Powered Exam & Interview Integrity Detection System

Offline Analysis · Audio + Video Integrity · Object/Gaze/Face Detection · Speaker Consistency

This project is an AI-driven integrity analysis system that processes pre-recorded video + audio to detect suspicious behavior during online exams or interview sessions.
It generates a rich, visual Session Integrity Report with:

🧑‍💻 Face presence tracking

👀 Eye/gaze direction analysis

📱📚 Forbidden object detection (phone, book, etc.)

🗣️ Audio speaker-consistency scoring (WavLM model)

📊 Timeline charts

⭐ Overall integrity score (0–100)

All analysis runs offline on the client machine — no data leaves the system.

🚀 Features
🎞️ Video Analysis

Face detection using MTCNN (facenet-pytorch)

Eye tracking using MediaPipe FaceMesh

Gaze direction classification (left, right, center, up, down)

Blink & EAR (Eye Aspect Ratio) tracking

Face-missing alerts (user away from screen)

Multi-face detection (extra persons in frame)

📦 Object Detection

YOLOv8-Nano (lightweight Ultralytics model)

Detects:

Mobile phones

Books / notes

(Easily extendable to more objects)

FPS-aware throttling for real-time efficiency

🔊 Audio Integrity Analysis

Extracts audio using ffmpeg

Uses WavLM-Base+ Speaker Verification Model

Splits audio into 2–3 second chunks

Computes embeddings for every chunk

Measures:

Average similarity

Minimum chunk similarity

Speaker change likelihood

Overall speaker consistency score (0–1 → 0–100)

📑 Session Integrity Report

Alerts summary (by type, by minute)

Timeline visualization using Chart.js

Speaker consistency visualization

Video score, audio score, and combined score

Stored as JSON per session

Rendered on a Flask dashboard

🧱 Project Architecture
exam-cheating-detection-main/
│
├── src/
│   ├── dashboard/           # Flask web UI
│   │   ├── app.py           # Upload, routing, report pages
│   │   └── templates/       # dashboard.html, upload.html, session_report.html
│   │
│   ├── detection/           # All detection modules
│   │   ├── face_detection.py
│   │   ├── eye_tracking.py
│   │   ├── object_detection.py
│   │   └── multi_face.py
│   │
│   ├── audio/               # Audio analysis pipeline
│   │   ├── speaker_consistency.py
│   │   └── utils_audio.py
│   │
│   ├── analysis/            # Scoring logic
│   │   ├── scoring.py       # Audio + video + combined scoring
│   │   └── report_generator.py
│   │
│   ├── utils/
│   │   ├── logging.py       # Alert logging
│   │   ├── screenshot_utils.py
│   │   └── timer.py
│   │
│   ├── offline_processor.py # Main offline pipeline
│   └── config.yaml          # All detection parameters
│
├── uploads/                 # Uploaded audio/video files
├── logs/
│   └── sessions/            # Stored JSON reports
│
├── requirements.txt
└── README.md                # ← You are here

⚙️ Installation
1️⃣ Create Conda environment (Python 3.10 is required)
conda create -n interview310 python=3.10
conda activate interview310

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install ffmpeg (required for audio extraction)

Mac:

brew install ffmpeg

🧪 Running the App

Start the Flask dashboard:

python -m src.dashboard.app


Navigate to:

http://127.0.0.1:5000

📤 Using the System

Upload pre-recorded video

Upload corresponding audio file (same duration recommended)

Click Analyze Recording

Wait for processing (15 min video ≈ 1–2 min offline processing)

View detailed Session Integrity Report

📊 Scoring System

We compute three scores:

1. 🎞️ Video Integrity Score (0–100)

Penalizes:

Face missing

Gaze away (left/right/up/down)

Excessive eye movement

Multiple faces

Forbidden object detection

Weighted formula maps alerts/minute → score.

2. 🔊 Audio Integrity Score (0–100)

Computed using WavLM embeddings:

High average similarity → high integrity

Large similarity dips → potential speaker change

speaker_change_flag = True → score penalty

Also robust to:

No audio

Corrupt file

Low activity audio

3. ⭐ Overall Integrity Score
overall = 0.7 * video_score + 0.3 * audio_score


Weight can be adjusted in scoring.py.

🧠 Models Used
Task	Model	Framework
Face Detection	MTCNN	facenet-pytorch
Eye/Gaze Tracking	FaceMesh	MediaPipe
Object Detection	YOLOv8n	Ultralytics
Speaker Embeddings	WavLM-Base+	HuggingFace Transformers
Audio Extraction	ffmpeg	subprocess

All models are pre-trained, so no fine-tuning needed and processing is efficient.

🧩 Configuration

Modify detection parameters in:

src/config.yaml


Examples:

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

🧼 Code Quality Improvements Implemented

Unified scoring pipeline

Defensive JSON schema handling

Robust Jinja templating (handling dict/float audio)

Optimized YOLO inference (resize + FPS throttle)

Easier debugging (python -m src.dashboard.app)

Fully isolated audio module (audio/speaker_consistency.py)

Environment fixes (transformers + tokenizers compatibility)

Removed .venv conflicts in favor of single conda environment

🛡️ Privacy & Local-Only Guarantee

This system performs all processing offline.
No recordings, audio, or metadata is uploaded to any external server.

📬 Future Enhancements

Real-time detection mode

Emotion detection

OCR-based note detection

Better multi-speaker diarization

Deepfake voice detection

Cloud deployment (FastAPI + GPU support)