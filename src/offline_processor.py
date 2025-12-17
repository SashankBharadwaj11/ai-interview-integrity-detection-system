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
from .analysis.alert_severity import compute_severity 

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

        self.output_cfg = self.cfg.get("output", {})
        self.annotated_dir = Path(
            self.output_cfg.get("annotated_video_dir", "./logs/annotated_videos")
        )
        self.annotated_dir.mkdir(parents=True, exist_ok=True)

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
    # MAIN ENTRY: analyze a prerecorded video
    # -------------------------------------------------------------
    def analyze_video(self, video_path: str, audio_path: Optional[str] = None):
        """
        Offline analysis of a pre-recorded video (+ optional separate audio file).

        - Runs all video detectors (face, gaze, multi-face, objects, mouth)
        - Optionally analyzes audio with SpeakerConsistencyAnalyzer
        - Writes an annotated video with overlays
        - Logs alerts & scores to a JSON file
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        session_id = self._generate_session_id(video_path)
        start_time = datetime.now()

        # --- optional audio analysis (speaker consistency) ---
        audio_summary = None
        if audio_path:
            try:
                speaker_analyzer = SpeakerConsistencyAnalyzer()
                audio_summary = speaker_analyzer.analyze(audio_path)
            except Exception as e:
                audio_summary = {"error": str(e)}

        # --- annotated video writer ---
        save_annotated = bool(self.output_cfg.get("save_annotated_video", True))
        writer = None
        annotated_path = None

        if save_annotated:
            fourcc = cv2.VideoWriter_fourcc(
                *self.output_cfg.get("annotated_codec", "mp4v")
            )
            annotated_path = str(self.annotated_dir / f"{session_id}.mp4")
            writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        # --- main loop ---
        frame_idx = 0
        alerts = []
        face_missing_start = None
        gaze_away_start = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp_s = frame_idx / fps

            # --- run detectors ---
            face_present = self.face_detector.detect_face(frame)
            # Handle both 2-value and 3-value returns from EyeTracker
            eye_res = self.eye_tracker.track_eyes(frame)
            if isinstance(eye_res, tuple) and len(eye_res) == 3:
                gaze_direction, eye_ratio, gaze_vec = eye_res
            else:
                # Legacy behavior: (gaze_direction, eye_ratio)
                gaze_direction, eye_ratio = eye_res
                gaze_vec = None
                
            mouth_moving = self.mouth_monitor.monitor_mouth(frame)
            objects_detected, object_list = self.object_detector.detect_objects(frame)
            
            # Handle both possible signatures of multi_face_detector:
            #   - returns bool
            #   - or returns (bool, num_faces)
            # Multi-face detector: now returns (multiple_faces, num_faces, face_boxes)
            mf_result = self.multi_face_detector.detect_multiple_faces(frame)
            if isinstance(mf_result, tuple) and len(mf_result) == 3:
                multiple_faces, num_faces, face_boxes = mf_result
            else:
                # backward-compat fallback
                multiple_faces = bool(mf_result)
                num_faces = 1 if face_present else 0
                face_boxes = []

            # --- build contextual state for this frame ---
            frame_ctx = {
                "face_present": face_present,
                "gaze_direction": gaze_direction,
                "eye_ratio": eye_ratio,
                "gaze_vector": gaze_vec,    
                "mouth_moving": mouth_moving,
                "objects": object_list,
                "multiple_faces": multiple_faces,
                "num_faces": num_faces,
                "face_boxes": face_boxes,     
            }

            frame_alerts = self._generate_alerts(timestamp_s, frame_ctx)

            if writer is not None:
                self._draw_overlays(frame, frame_idx, timestamp_s, frame_alerts, frame_ctx)
                writer.write(frame)

            # track durations for behavioral alerts
            # FACE_MISSING duration
            if not face_present:
                if face_missing_start is None:
                    face_missing_start = timestamp_s
                face_missing_duration = timestamp_s - face_missing_start
            else:
                face_missing_duration = 0.0
                face_missing_start = None

            # GAZE_AWAY duration
            if gaze_direction in ("left", "right", "up", "down"):
                if gaze_away_start is None:
                    gaze_away_start = timestamp_s
                gaze_away_duration = timestamp_s - gaze_away_start
            else:
                gaze_away_duration = 0.0
                gaze_away_start = None

            frame_ctx["face_missing_duration"] = face_missing_duration
            frame_ctx["gaze_away_duration"] = gaze_away_duration

            # --- generate alerts with meaningful severity/details ---
            frame_alerts = []

            if not face_present and face_missing_duration >= 1.0:
                details = {"duration_sec": face_missing_duration}
                frame_alerts.append({
                    "timestamp": timestamp_s,
                    "type": "FACE_MISSING",
                    "severity": compute_severity("FACE_MISSING", details),
                    "details": details,
                })

            if gaze_direction in ("left", "right", "up", "down") and gaze_away_duration >= 1.0:
                details = {
                    "direction": gaze_direction,
                    "eye_ratio": eye_ratio,
                    "duration_sec": gaze_away_duration,
                }
                frame_alerts.append({
                    "timestamp": timestamp_s,
                    "type": "GAZE_AWAY",
                    "severity": compute_severity("GAZE_AWAY", details),
                    "details": details,
                })

            if multiple_faces and num_faces >= 2:
                details = {"num_faces": num_faces}
                frame_alerts.append({
                    "timestamp": timestamp_s,
                    "type": "MULTI_FACE",
                    "severity": compute_severity("MULTI_FACE", details),
                    "details": details,
                })

            if objects_detected:
                for obj in object_list:
                    details = {
                        "label": obj["label"],
                        "confidence": obj["confidence"],
                        "bbox": obj["bbox"],
                    }
                    frame_alerts.append({
                        "timestamp": timestamp_s,
                        "type": "OBJECT_DETECTED",
                        "severity": compute_severity("OBJECT_DETECTED", details),
                        "details": details,
                    })

            # append to global alerts
            alerts.extend(frame_alerts)

            # --- write annotated frame if enabled ---
            if writer is not None:
                self._draw_overlays(frame, frame_idx, timestamp_s, frame_alerts, frame_ctx)
                writer.write(frame)

        cap.release()
        if writer is not None:
            writer.release()

        end_time = datetime.now()
        duration_sec = frame_idx / fps if fps > 0 else 0.0

        # --- build session summary dataclass / dict ---
        summary = {
            "session_id": session_id,
            "duration_seconds": duration_sec,
            "num_frames": frame_idx,
            "fps": fps,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "annotated_video_path": annotated_path,
        }

        # --- scoring ---
        from collections import Counter
        type_counts = Counter(a["type"] for a in alerts)
        total_alerts = len(alerts)
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0

        summary_stats = {
            "total_alerts": total_alerts,
            "by_type": dict(type_counts),
            "alerts_per_minute": total_alerts / duration_min if duration_min > 0 else 0.0,
            "duration_seconds": duration_sec,
        }

        video_score = compute_video_score(summary_stats, alerts)
        audio_score = compute_audio_score(audio_summary)
        overall_score = compute_overall_score(video_score, audio_score)

        scores = {
            "video": video_score,
            "audio": audio_score,
            "overall": overall_score,
        }

        # --- persist JSON log for dashboard ---
        session_log = {
            "session": summary,
            "summary": summary_stats,
            "alerts": alerts,
            "scores": scores,
            "audio": audio_summary or {},
        }

        self._save_session_log_json(session_id, session_log)

        return session_log
    # -------------------------------------------------------------
    # Alert rule engine (high-level, config-driven)
    # -------------------------------------------------------------
    def _generate_alerts(self, timestamp_s: float, frame_ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate behavior-aware alerts for a single frame.

        Parameters
        ----------
        timestamp_s : float
            Time of this frame in seconds from start.
        frame_ctx : dict
            Per-frame context, expected keys (all optional but recommended):

            face_present: bool
            face_missing_duration: float seconds
            gaze_direction: 'left'|'right'|'up'|'down'|'center'
            eye_ratio: float (EAR)
            gaze_away_duration: float seconds
            multiple_faces: bool
            num_faces: int
            mouth_moving: bool
            objects: list of {label, confidence, bbox}

        Returns
        -------
        list[dict]
            Each dict has:
            {
              "timestamp": float,
              "type": str,
              "severity": str,
              "details": dict
            }
        """
        alerts: List[Dict[str, Any]] = []

        # ---------- thresholds from config ----------
        det_cfg = self.cfg.get("detection", {})

        # How long face must be missing before we alert (seconds)
        face_missing_min_sec = float(
            det_cfg.get("face", {}).get("missing_threshold_sec", 1.0)
        )

        # Gaze away threshold (seconds)
        gaze_cfg = det_cfg.get("eyes", {})
        gaze_min_sec = float(gaze_cfg.get("gaze_threshold", 2.0))

        # Multi-face: minimum number of faces to consider a violation
        multi_face_cfg = det_cfg.get("multi_face", {})
        multi_face_min_faces = int(multi_face_cfg.get("min_faces", 2))

        # ---------- extract context safely ----------
        face_present = bool(frame_ctx.get("face_present", False))
        face_missing_duration = float(frame_ctx.get("face_missing_duration", 0.0))

        gaze_direction = frame_ctx.get("gaze_direction", "center")
        eye_ratio = float(frame_ctx.get("eye_ratio", 0.0))
        gaze_away_duration = float(frame_ctx.get("gaze_away_duration", 0.0))

        multiple_faces = bool(frame_ctx.get("multiple_faces", False))
        num_faces = int(frame_ctx.get("num_faces", 0))

        mouth_moving = bool(frame_ctx.get("mouth_moving", False))

        objects = frame_ctx.get("objects", []) or []

        # ---------- FACE_MISSING ----------
        if (not face_present) and (face_missing_duration >= face_missing_min_sec):
            details = {
                "duration_sec": face_missing_duration,
            }
            alerts.append({
                "timestamp": timestamp_s,
                "type": "FACE_MISSING",
                "severity": compute_severity("FACE_MISSING", details),
                "details": details,
            })

        # ---------- GAZE_AWAY ----------
        if gaze_direction in ("left", "right", "up", "down") and gaze_away_duration >= gaze_min_sec:
            details = {
                "direction": gaze_direction,
                "eye_ratio": eye_ratio,
                "duration_sec": gaze_away_duration,
            }
            alerts.append({
                "timestamp": timestamp_s,
                "type": "GAZE_AWAY",
                "severity": compute_severity("GAZE_AWAY", details),
                "details": details,
            })

        # ---------- MULTI_FACE ----------
        if multiple_faces and num_faces >= multi_face_min_faces:
            details = {
                "num_faces": num_faces,
            }
            alerts.append({
                "timestamp": timestamp_s,
                "type": "MULTI_FACE",
                "severity": compute_severity("MULTI_FACE", details),
                "details": details,
            })

        # ---------- MOUTH_MOVEMENT ----------
        # You can tune mouth_moving logic further, but this at least surfaces it.
        if mouth_moving:
            details = {
                "note": "Mouth movement detected (possible talking)",
            }
            alerts.append({
                "timestamp": timestamp_s,
                "type": "MOUTH_MOVEMENT",
                "severity": compute_severity("MOUTH_MOVEMENT", details),
                "details": details,
            })

        # ---------- OBJECT_DETECTED ----------
        for obj in objects:
            label = obj.get("label", "")
            conf = float(obj.get("confidence", 0.0))
            bbox = obj.get("bbox", None)

            details = {
                "label": label,
                "confidence": conf,
            }
            if bbox is not None:
                details["bbox"] = bbox

            alerts.append({
                "timestamp": timestamp_s,
                "type": "OBJECT_DETECTED",
                "severity": compute_severity("OBJECT_DETECTED", details),
                "details": details,
            })

        return alerts
    
    def _draw_overlays(self, frame, frame_idx: int, timestamp: float,
                       frame_alerts: list, ctx: dict) -> None:
        """
        Draws diagnostic overlays on the frame:
        - Frame index & timestamp
        - Gaze direction, EAR, num faces
        - Alert types/severity
        - Face boxes (blue for main user, red for others)
        - YOLO object boxes (phone/book/etc.)
        - Gaze vector arrow
        """
        h, w = frame.shape[:2]
        y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        fg = (255, 255, 255)
        bg = (0, 0, 0)

        def put(text: str):
            nonlocal y
            cv2.putText(frame, text, (10, y + 1), font, scale, bg, 3, cv2.LINE_AA)
            cv2.putText(frame, text, (10, y),     font, scale, fg, 1, cv2.LINE_AA)
            y += 22

        # Basic text info
        put(f"Frame: {frame_idx}  t={timestamp:.2f}s")

        if ctx.get("gaze_direction") is not None:
            put(f"Gaze: {ctx['gaze_direction']}  EAR={ctx.get('eye_ratio', 0):.3f}")

        if ctx.get("num_faces") is not None:
            put(f"Faces: {ctx['num_faces']}")

        # Show alerts for this frame
        for a in frame_alerts:
            t = a.get("type", "")
            sev = a.get("severity", "")
            put(f"ALERT: {t} [{sev}]")

        # ----------------------------------------
        # Face bounding boxes (BLUE = main user, RED = others)
        # ----------------------------------------
        face_boxes = ctx.get("face_boxes", []) or []
        for i, box in enumerate(face_boxes):
            x1, y1, x2, y2 = map(int, box)

            if i == 0:
                color = (255, 0, 0)     # BLUE in BGR? (0,0,255) is red; (255,0,0) is blue
                label = "User"
            else:
                color = (0, 0, 255)     # RED
                label = f"Other #{i}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(15, y1 - 5)),
                font,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------
        # YOLO object boxes: phone/book/etc.
        # ----------------------------------------
        objects = ctx.get("objects") or []
        for obj in objects:
            label = obj.get("label", "")
            conf = float(obj.get("confidence", 0.0))
            bbox = obj.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(int, bbox)

            # color by object type
            if label in ("cell phone", "phone", "mobile"):
                color = (0, 0, 255)      # red
            elif label in ("book", "notebook"):
                color = (0, 255, 255)    # yellow
            else:
                color = (0, 255, 0)      # green

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, max(15, y1 - 5)),
                        font, 0.5, color, 2, cv2.LINE_AA)

        # ----------------------------------------
        # Gaze vector arrow
        # ----------------------------------------
        gaze_vec = ctx.get("gaze_vector")
        if gaze_vec is not None:
            eye_center = gaze_vec.get("eye_center")
            direction = gaze_vec.get("direction", "center")

            if eye_center is not None:
                ex, ey = int(eye_center[0]), int(eye_center[1])
                # small dot at eye center
                cv2.circle(frame, (ex, ey), 4, (255, 255, 0), -1)

                length = 80
                dx, dy = 0, 0
                if direction == "left":
                    dx = -length
                elif direction == "right":
                    dx = length
                elif direction == "up":
                    dy = -length
                elif direction == "down":
                    dy = length

                end_point = (ex + dx, ey + dy)
                cv2.arrowedLine(
                    frame,
                    (ex, ey),
                    end_point,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
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

    def _save_session_log_json(self, session_id: str, data: dict) -> None:
        out_path = self.logs_dir / f"{session_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)


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
    )
