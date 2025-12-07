import cv2
import torch
from ultralytics import YOLO
from datetime import datetime


class ObjectDetector:
    def __init__(self, config):
        """
        config: full global config dict (not just objects)
        expects: config['detection']['objects'] with keys:
            - min_confidence
            - detection_interval
            - max_fps
        """
        objects_cfg = config['detection']['objects']
        self.config = objects_cfg

        # COCO class IDs for interest
        self.class_map = {
            73: 'book',
            67: 'cell phone',
        }

        self.alert_logger = None

        self.detection_interval = objects_cfg.get('detection_interval', 5)
        self.max_fps = objects_cfg.get('max_fps', 5)
        self.frame_count = 0
        self.last_detection_time = datetime.min

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._initialize_model()

    def _initialize_model(self):
        """Initialize optimized YOLO model."""
        try:
            # smallest YOLOv8 model for speed
            self.model = YOLO('models/yolov8n.pt')

            # Hyperparameters / overrides
            self.model.overrides['conf'] = self.config.get('min_confidence', 0.65)
            self.model.overrides['iou'] = 0.45
            self.model.overrides['imgsz'] = 320

            # Move model to device
            self.model.to(self.device)

            # Warm up
            dummy_input = torch.zeros((1, 3, 320, 320), device=self.device)
            self.model(dummy_input)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize object detector: {str(e)}")

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def detect_objects(self, frame, visualize=False):
        """
        Returns:
            detected (bool): True if any forbidden object detected.
            objects (list[dict]): List of detections with label, confidence, bbox.
        """
        current_time = datetime.now()
        time_since_last = (current_time - self.last_detection_time).total_seconds()

        # FPS throttling
        if time_since_last < (1.0 / self.max_fps):
            return False, []

        # Frame-based skipping
        self.frame_count += 1
        if self.frame_count % self.detection_interval != 0:
            return False, []

        try:
            orig_h, orig_w = frame.shape[:2]
            new_w = 320
            new_h = int(orig_h * (new_w / orig_w))
            resized_frame = cv2.resize(frame, (new_w, new_h))

            # Run YOLO
            results = self.model(resized_frame, verbose=False)

            detected = False
            objects = []

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)

                    if cls in self.class_map and conf > self.config.get('min_confidence', 0.65):
                        detected = True
                        label = self.class_map[cls]

                        # Scale back to original size
                        x1, y1, x2, y2 = box.xyxy[0]
                        sx = orig_w / new_w
                        sy = orig_h / new_h
                        x1 = int(x1 * sx)
                        y1 = int(y1 * sy)
                        x2 = int(x2 * sx)
                        y2 = int(y2 * sy)

                        det = {
                            "label": label,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                        objects.append(det)

                        if self.alert_logger:
                            self.alert_logger.log_alert(
                                "FORBIDDEN_OBJECT",
                                f"Detected {label} with confidence {conf:.2f}"
                            )

                        if visualize:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            self.last_detection_time = current_time
            return detected, objects

        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "OBJECT_DETECTION_ERROR",
                    f"Object detection failed: {str(e)}"
                )
            return False, []