from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from uuid import uuid4
import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR) + "/src")

from ..offline_processor import analyze_recording  # adjust import to your structure

app = Flask(__name__)
app.secret_key = "change-me"

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
UPLOAD_DIR = BASE_DIR / "uploads"
LOGS_DIR = BASE_DIR / "logs" / "sessions"
UPLOAD_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}
ALLOWED_AUDIO_EXT = {"wav", "mp3", "m4a"}


def allowed_file(filename, allowed_exts):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts


# ---------- NEW DASHBOARD ROUTE ----------

@app.route("/")
def dashboard():
    """
    Home page. Lists all completed sessions (from logs/*.json)
    and shows a button to upload a new prerecorded recording.
    """
    sessions = []

    for path in sorted(LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(path, "r") as f:
            data = json.load(f)
        ses = data.get("session", {})
        # ensure session_id is present
        sid = ses.get("session_id") or path.stem
        ses["session_id"] = sid
        sessions.append(ses)

    return render_template("dashboard.html", sessions=sessions)


# ---------- UPLOAD ROUTE (from previous step) ----------

@app.route("/upload", methods=["GET", "POST"])
def upload_recording():
    print(">>> ENTER upload_recording")
    if request.method == "POST":
        video_file = request.files.get("video")
        audio_file = request.files.get("audio")

        if not video_file or not audio_file:
            flash("Please upload both video and audio files")
            return redirect(request.url)

        if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXT):
            flash("Invalid video format")
            return redirect(request.url)
 
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXT):
            flash("Invalid audio format")
            return redirect(request.url)

        session_id = uuid4().hex
        video_filename = secure_filename(f"{session_id}_video_{video_file.filename}")
        audio_filename = secure_filename(f"{session_id}_audio_{audio_file.filename}")

        video_path = UPLOAD_DIR / video_filename
        audio_path = UPLOAD_DIR / audio_filename

        video_file.save(video_path)
        audio_file.save(audio_path)

        session_log = analyze_recording(
            video_path=str(video_path),
            audio_path=str(audio_path),
            session_id=session_id,
            config_path=str(BASE_DIR / "config" / "config.yaml"),
            logs_dir=str(LOGS_DIR),
        )

        session_info = session_log.get("session", {})
        sid = session_info.get("session_id", session_id)
        num_alerts = len(session_log.get("alerts", []))

        flash(f"Processed session {sid} with {num_alerts} alerts")
        return redirect(url_for("session_report", session_id=sid))

    return render_template("upload.html")



# ---------- SESSION REPORT ROUTE ----------

@app.route("/session/<session_id>")
def session_report(session_id):
    log_path = LOGS_DIR / f"{session_id}.json"
    if not log_path.exists():
        return f"No report found for session {session_id}", 404

    with open(log_path, "r") as f:
        data = json.load(f)

    return render_template(
        "session_report.html",
        session=data["session"],
        summary=data["summary"],
        alerts=data["alerts"],
        scores=data.get("scores", {}),
        audio=data.get("audio", {}),
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)

# from flask import Flask, render_template, request, redirect, url_for, flash
# from werkzeug.utils import secure_filename
# from uuid import uuid4
# from pathlib import Path
# import sys
# import yaml,os
# from datetime import datetime
# from flask import jsonify
# import json

# BASE_DIR = Path(__file__).resolve().parents[2]
# if str(BASE_DIR) not in sys.path:
#     sys.path.insert(0, str(BASE_DIR) + "/src")

# from offline_processor import analyze_recording  # adjust import to your structure

# app = Flask(__name__)
# app.secret_key = "change-me"

# UPLOAD_DIR = BASE_DIR / "uploads"
# LOGS_DIR = BASE_DIR / "logs"
# UPLOAD_DIR.mkdir(exist_ok=True)
# LOGS_DIR.mkdir(exist_ok=True)

# ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}
# ALLOWED_AUDIO_EXT = {"wav", "mp3", "m4a"}

# # Load configuration
# with open('config/config.yaml') as f:
#     config = yaml.safe_load(f)

# def allowed_file(filename, allowed_exts):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts

# @app.route('/')
# def dashboard():
#     return render_template('dashboard.html')

# @app.route('/api/alerts')
# def get_alerts():
#     log_file = os.path.join(config['logging']['log_path'], "alerts.log")
#     alerts = []
    
#     if os.path.exists(log_file):
#         with open(log_file, 'r') as f:
#             alerts = [line.strip() for line in f.readlines()[-10:]]  # Get last 10 alerts
            
#     return jsonify(alerts)

# @app.route('/api/stats')
# def get_stats():
#     # This would be more sophisticated in a real implementation
#     return jsonify({
#         'face_detected': True,
#         'current_activity': 'Normal',
#         'cheating_probability': 15,
#         'last_alert': datetime.now().strftime("%H:%M:%S")
#     })

# @app.route("/upload", methods=["GET", "POST"])
# def upload_recording():
#     if request.method == "POST":
#         video_file = request.files.get("video")
#         audio_file = request.files.get("audio")

#         if not video_file or not audio_file:
#             flash("Please upload both video and audio files")
#             return redirect(request.url)

#         if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXT):
#             flash("Invalid video format")
#             return redirect(request.url)

#         if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXT):
#             flash("Invalid audio format")
#             return redirect(request.url)

#         # Save files
#         session_id = uuid4().hex
#         video_filename = secure_filename(f"{session_id}_video_{video_file.filename}")
#         audio_filename = secure_filename(f"{session_id}_audio_{audio_file.filename}")

#         video_path = UPLOAD_DIR / video_filename
#         audio_path = UPLOAD_DIR / audio_filename

#         video_file.save(video_path)
#         audio_file.save(audio_path)

#         # Run offline analysis (blocking call here – fine for now)
#         summary = analyze_recording(
#             video_path=str(video_path),
#             audio_path=str(audio_path),
#             session_id=session_id,
#             config_path=str(BASE_DIR / "config" / "config.yaml"),
#             logs_dir=str(LOGS_DIR),
#         )

#         flash(f"Processed session {summary.session_id} with {len(summary.alerts)} alerts")
#         return redirect(url_for("session_report", session_id=session_id))

#     # GET => show upload form
#     return render_template("upload.html")

# @app.route("/session/<session_id>")
# def session_report(session_id):
#     log_path = LOGS_DIR / f"{session_id}.json"
#     if not log_path.exists():
#         return f"Session {session_id} not found", 404

#     with open(log_path, "r") as f:
#         data = json.load(f)

#     session = data["session"]
#     alerts = data["alerts"]

#     # You can compute some aggregates for nicer UI
#     type_counts = {}
#     for a in alerts:
#         type_counts[a["type"]] = type_counts.get(a["type"], 0) + 1

#     return render_template(
#         "session_report.html",
#         session=session,
#         alerts=alerts,
#         type_counts=type_counts,
#     )

# if __name__ == '__main__':
#     app.run(debug=True)