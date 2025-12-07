import cv2
import yaml
import json
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Reuse existing detectors (no CV logic here) ---
from .detection.face_detection import FaceDetector
from .detection.eye_tracking import EyeTracker
from .detection.mouth_detection import MouthMonitor
from .detection.object_detection import ObjectDetector
from .detection.multi_face import MultiFaceDetector
# NOTE: we still import AudioMonitor but we won't use it (dummy skip)
from .detection.audio_detection import AudioMonitor  # noqa: F401
from .audio.speaker_consistency import SpeakerConsistencyAnalyzer


# --- Optional utilities already in the repo (used only for logging / reports) ---
from .utils.logging import AlertLogger
from .utils.violation_logger import ViolationLogger
from .utils.screenshot_utils import ViolationCapturer
from .reporting.report_generator import ReportGenerator

from .analysis.scoring import (
    compute_video_score,
    compute_audio_score,
    compute_overall_score,
)

# Offline speed parameters
FRAME_STRIDE = 3          # process 1 of every 3 frames
RESIZE_WIDTH = 640        # downscale to 640px wide
OBJ_DETECT_EVERY = 5      # run object detector every 5 processed frames
MF_DETECT_EVERY = 2       # run multi-face detection every 2 processed frames



# ---------------------------------------------------------------------
# Data structures for clean summary + dashboard integration
# ---------------------------------------------------------------------

@dataclass
class AlertEvent:
    session_id: str
    timestamp: float           # seconds from start of video
    frame_index: int
    type: str                  # e.g. FACE_MISSING, GAZE_AWAY, MULTI_FACE, OBJECT_DETECTED
    severity: str              # "low" | "medium" | "high"
    details: Dict[str, Any]


@dataclass
class SessionSummary:
    session_id: str
    duration_seconds: float
    num_frames: int
    fps: float
    alerts: List[AlertEvent]


# ---------------------------------------------------------------------
# Helper to load config
# ---------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Core offline analyzer – only orchestrates detectors and rules
# ---------------------------------------------------------------------

class OfflineExamAnalyzer:
    """
    Offline analyzer for prerecorded exam sessions.

    - Takes a video file (webcam recording).
    - Optionally takes an audio file path BUT IGNORES IT (dummy skip as requested).
    - Uses existing detection modules to compute per-frame signals.
    - Converts those signals into high-level alerts using thresholds from config.
    - Logs alerts to JSON so the dashboard can render reports.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        logs_dir: str = "logs",
        screenshots_dir: str = "logs/violations",
    ) -> None:
        
        self.cfg = load_config(config_path)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # --- detectors expect the FULL config dict (they access config['detection'][...]) ---
        self.face_detector = FaceDetector(self.cfg)
        self.eye_tracker = EyeTracker(self.cfg)
        self.mouth_monitor = MouthMonitor(self.cfg)
        self.object_detector = ObjectDetector(self.cfg)
        self.multi_face_detector = MultiFaceDetector(self.cfg)
        # AudioMonitor is imported but not used (dummy skip in offline mode)
        self.audio_monitor = AudioMonitor(self.cfg)


        # Logging / reporting utilities (already in repo)
        self.alert_logger = None
        self.violation_logger = None
        self.violation_capturer = None
        self.report_generator = ReportGenerator(self.cfg)

        # audio analyzer
        audio_tmp_dir = self.logs_dir / "audio_tmp"
        self.speaker_analyzer = SpeakerConsistencyAnalyzer(tmp_dir=audio_tmp_dir)

        # stateful counters for "consecutive frames" rules
        self._gaze_away_frames = 0
        self._mouth_moving_frames = 0
        self._multi_face_frames = 0

    # -------------------------------------------------------------
    # MAIN ENTRY: analyze a prerecorded video (audio is dummy)
    # -------------------------------------------------------------
    def analyze_video(
        self,
        video_path: str,
        session_id: Optional[str] = None,
        audio_path: Optional[str] = None,   # accepted but ignored
    ) -> SessionSummary:
        """
        Run offline analysis on a prerecorded webcam video.

        :param video_path: path to video file recorded during exam.
        :param session_id: optional explicit session id; if None, derived from timestamp.
        :param audio_path: path to audio file (ignored – dummy skip).
        """
        video_path = str(video_path)
        session_id = session_id or self._generate_session_id(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg.get("video", {}).get("fps", 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        alerts: List[AlertEvent] = []

        frame_idx = 0
        processed_frame_idx = 0  # count only frames we actually analyze
        last_objects_detected = False
        last_object_list: List[Dict[str, Any]] = []
        last_multiple_faces = False
        last_num_faces = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # timestamp based on original frame index (not just processed frames)
            t = frame_idx / fps if fps > 0 else 0.0

            # --- Frame skipping: process only every Nth frame ---
            if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE != 0):
                frame_idx += 1
                continue

            # --- Resize frame for much faster detection ---
            if RESIZE_WIDTH is not None:
                h, w = frame.shape[:2]
                if w > RESIZE_WIDTH:
                    scale = RESIZE_WIDTH / float(w)
                    new_size = (RESIZE_WIDTH, int(h * scale))
                    frame_small = cv2.resize(frame, new_size)
                else:
                    frame_small = frame
            else:
                frame_small = frame

            # --- 1. Call detectors on the smaller frame ---

            # Face detection
            face_present = self.face_detector.detect_face(frame_small)

            # Eye tracking
            gaze_direction, eye_ratio = self.eye_tracker.track_eyes(frame_small)

            # Mouth movement
            mouth_moving = self.mouth_monitor.monitor_mouth(frame_small)

            # Object detection: run only every OBJ_DETECT_EVERY processed frames
            if OBJ_DETECT_EVERY > 1 and (processed_frame_idx % OBJ_DETECT_EVERY != 0):
                objects_detected = last_objects_detected
                object_list = last_object_list
            else:
                obj_result = self.object_detector.detect_objects(frame_small)
                if isinstance(obj_result, tuple) and len(obj_result) == 2:
                    objects_detected, object_list = obj_result
                elif isinstance(obj_result, list):
                    object_list = obj_result
                    objects_detected = len(object_list) > 0
                else:
                    objects_detected = bool(obj_result)
                    object_list = []

                last_objects_detected = objects_detected
                last_object_list = object_list

            # Multi-face detection: run only every MF_DETECT_EVERY processed frames
            if MF_DETECT_EVERY > 1 and (processed_frame_idx % MF_DETECT_EVERY != 0):
                multiple_faces = last_multiple_faces
                num_faces = last_num_faces
            else:
                mf_result = self.multi_face_detector.detect_multiple_faces(frame_small)
                if isinstance(mf_result, tuple) and len(mf_result) == 2:
                    multiple_faces, num_faces = mf_result
                elif isinstance(mf_result, (int, float)):
                    num_faces = int(mf_result)
                    multiple_faces = num_faces > 1
                else:
                    multiple_faces = bool(mf_result)
                    num_faces = 2 if multiple_faces else 1

                last_multiple_faces = multiple_faces
                last_num_faces = num_faces

            # --- 2. Convert detector outputs into alerts ---
            frame_alerts = self._generate_alerts(
                session_id=session_id,
                frame_idx=frame_idx,        # original frame index
                t=t,
                face_present=face_present,
                gaze_direction=gaze_direction,
                eye_ratio=eye_ratio,
                mouth_moving=mouth_moving,
                multiple_faces=multiple_faces,
                num_faces=num_faces,
                objects_detected=objects_detected,
                object_list=object_list,
                frame=frame_small,          # we pass the resized frame to violation logging
            )

            alerts.extend(frame_alerts)
            processed_frame_idx += 1
            frame_idx += 1

        cap.release()

        duration_seconds = frame_idx / fps if fps > 0 else 0.0
        summary = SessionSummary(
            session_id=session_id,
            duration_seconds=duration_seconds,
            num_frames=frame_idx,
            fps=fps,
            alerts=alerts,
        )

        # --- audio analysis (if audio_path provided) ---
        audio_summary = None
        if audio_path:
            try:
                audio_summary = self.speaker_analyzer.analyze(audio_path)
            except Exception as e:
                audio_summary = {
                    "error": f"Audio analysis failed: {str(e)}",
                    "speaker_consistency_score": None,
                }

        self._save_session_log(summary, audio_summary=audio_summary)

        # --- 4. Generate PDF / HTML report (uses existing reporting module) ---
        try:
            self.report_generator.generate_offline_report_from_session_log(session_id)
        except Exception as e:
            # don't hard fail just because reporting failed
            print(f"[WARN] Failed to generate report for {session_id}: {e}")

        return summary

    # -------------------------------------------------------------
    # Alert rule engine (high-level, config-driven)
    # -------------------------------------------------------------
    def _generate_alerts(
        self,
        session_id: str,
        frame_idx: int,
        t: float,
        face_present: bool,
        gaze_direction: str,
        eye_ratio: float,
        mouth_moving: bool,
        multiple_faces: bool,
        num_faces: int,
        objects_detected: bool,
        object_list: List[Dict[str, Any]],
        frame,
    ) -> List[AlertEvent]:
        det_cfg = self.cfg.get("detection", {})
        eyes_cfg = det_cfg.get("eyes", {})
        mouth_cfg = det_cfg.get("mouth", {})
        multi_face_cfg = det_cfg.get("multi_face", {})
        objects_cfg = det_cfg.get("objects", {})

        alerts: List[AlertEvent] = []

        # --- Face missing ---
        if not face_present:
            evt = AlertEvent(
                session_id=session_id,
                timestamp=t,
                frame_index=frame_idx,
                type="FACE_MISSING",
                severity="medium",
                details={},
            )
            alerts.append(evt)
            self._log_violation("FACE_MISSING", evt, frame)

        # --- Gaze away / eye ratio logic ---
        # Example rule: if gaze_direction != "Center" for N consecutive frames, raise alert.
        gaze_consec_frames = eyes_cfg.get("consecutive_frames", 3)
        # Normalize gaze direction from detector
        direction_norm = (gaze_direction or "").strip().lower()

        # Treat these as "looking at screen"
        center_like = {"center", "centre", "front"}

        if direction_norm not in center_like:
            self._gaze_away_frames += 1
        else:
            self._gaze_away_frames = 0

        if self._gaze_away_frames >= gaze_consec_frames:
            evt = AlertEvent(
                session_id=session_id,
                timestamp=t,
                frame_index=frame_idx,
                type="GAZE_AWAY",
                severity="medium",
                details={
                    "direction": direction_norm,
                    "eye_ratio": float(eye_ratio),
                },
            )
            alerts.append(evt)
            self._log_violation("GAZE_AWAY", evt, frame)
            self._gaze_away_frames = 0

        # --- Mouth movement ---
        if mouth_moving:
            self._mouth_moving_frames += 1
        else:
            self._mouth_moving_frames = 0

        mouth_thresh = mouth_cfg.get("movement_threshold", 3)
        if self._mouth_moving_frames >= mouth_thresh:
            evt = AlertEvent(
                session_id=session_id,
                timestamp=t,
                frame_index=frame_idx,
                type="MOUTH_MOVEMENT",
                severity="medium",
                details={},
            )
            alerts.append(evt)
            self._log_violation("MOUTH_MOVEMENT", evt, frame)
            self._mouth_moving_frames = 0

        # --- Multiple faces ---
        if multiple_faces:
            self._multi_face_frames += 1
        else:
            self._multi_face_frames = 0

        multi_face_thresh = multi_face_cfg.get("alert_threshold", 5)
        if self._multi_face_frames >= multi_face_thresh:
            evt = AlertEvent(
                session_id=session_id,
                timestamp=t,
                frame_index=frame_idx,
                type="MULTI_FACE",
                severity="high",
                details={"num_faces": int(num_faces)},
            )
            alerts.append(evt)
            self._log_violation("MULTI_FACE", evt, frame)
            self._multi_face_frames = 0

        # --- Prohibited objects (phone, book, etc.) ---
        if objects_detected and object_list:
            for obj in object_list:
                evt = AlertEvent(
                    session_id=session_id,
                    timestamp=t,
                    frame_index=frame_idx,
                    type="OBJECT_DETECTED",
                    severity="high",
                    details={
                        "label": obj.get("label"),
                        "confidence": float(obj.get("confidence", 0.0)),
                        "bbox": obj.get("bbox", []),
                    },
                )
                alerts.append(evt)
                self._log_violation("OBJECT_DETECTED", evt, frame)

        return alerts

    # -------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------
    def _log_violation(self, vtype: str, evt: AlertEvent, frame) -> None:
        """
        Wraps violation logging + (optional) screenshot capture.

        In OFFLINE mode:
        - We rely on the JSON session log created in _save_session_log()
        - ViolationLogger is not used, because its original signature is
            specific to the live streaming pipeline.
        """
        # Optionally, if you later decide to support ViolationLogger here,
        # add a guarded call matching its real signature.
        if self.violation_logger is not None:
            try:
                # TODO: adapt to ViolationLogger.log_violation(...) signature
                # self.violation_logger.log_violation(...)
                pass
            except Exception as e:
                print(f"[WARN] ViolationLogger call failed: {e}")

        # Capture screenshot frame for this violation (OFFLINE OPTIONAL)
        if self.violation_capturer is not None:
            try:
                self.violation_capturer.capture_violation(
                    frame=frame,
                    session_id=evt.session_id,
                    violation_type=vtype,
                    timestamp=evt.timestamp,
                )
            except Exception as e:
                print(f"[WARN] Failed to capture violation screenshot: {e}")

    def _save_session_log(self,summary: SessionSummary,audio_summary: Optional[Dict] = None,) -> None:
        """
        Save per-session JSON with:
          - session metadata
          - summary stats
          - full alert list
          - scores (video / audio / overall)
        """
        out_path = self.logs_dir / f"{summary.session_id}.json"

        # Convert alerts to plain dicts
        alerts_list = [asdict(a) for a in summary.alerts]

        from collections import Counter
        type_counts = Counter(a["type"] for a in alerts_list)
        total_alerts = len(alerts_list)
        duration_sec = summary.duration_seconds
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0

        summary_stats = {
            "total_alerts": total_alerts,
            "by_type": dict(type_counts),
            "alerts_per_minute": total_alerts / duration_min if duration_min > 0 else 0.0,
            "duration_seconds": duration_sec,
        }

        # --- scoring ---
        video_score = compute_video_score(summary_stats, alerts_list)
        audio_summary = compute_audio_score(audio_summary)
        audio_score = compute_audio_score(audio_summary)

        overall_score = compute_overall_score(video_score, audio_score)

        scores = {
            "video": video_score,
            "audio": audio_score,
            "overall": overall_score,
        }

        data = {
            "session": {
                "session_id": summary.session_id,
                "duration_seconds": summary.duration_seconds,
                "num_frames": summary.num_frames,
                "fps": summary.fps,
                "generated_at": datetime.now().isoformat(),
            },
            "summary": summary_stats,
            "alerts": alerts_list,
            "scores": scores,
            "audio": audio_summary,
        }

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        try:
            self.report_generator.generate_offline_report_from_session_log(
                str(out_path),
                output_format="html",
            )
        except Exception as e:
            print(f"[WARN] Failed to generate offline report: {e}")


    @staticmethod
    def _generate_session_id(video_path: str) -> str:
        basename = Path(video_path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{basename}_{ts}"


# ---------------------------------------------------------------------
# Simple functional wrapper to plug into Flask dashboard
# ---------------------------------------------------------------------

def analyze_recording(
    video_path: str,
    audio_path: Optional[str] = None,
    session_id: Optional[str] = None,
    config_path: str = "config/config.yaml",
    logs_dir: str = "logs",
) -> SessionSummary:
    """
    Convenience function for dashboard / CLI.

    NOTE: `audio_path` is accepted to keep the function signature
    stable, but it is intentionally **ignored** in this offline
    implementation (dummy skip).
    """
    analyzer = OfflineExamAnalyzer(config_path=config_path, logs_dir=logs_dir)
    return analyzer.analyze_video(
        video_path=video_path,
        audio_path=audio_path,    # currently unused
        session_id=session_id,
    )
