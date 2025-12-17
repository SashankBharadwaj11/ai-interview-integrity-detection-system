import cv2
import torch
from facenet_pytorch import MTCNN

class MultiFaceDetector:
    """
    Detects multiple faces in a frame using MTCNN and returns:
      - multiple_faces: bool
      - num_faces: int
      - face_boxes: list[[x1, y1, x2, y2], ...]
    """

    def __init__(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            post_process=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            device=self.device
        )
        # how many consecutive frames with multiple faces before alert
        self.threshold = config['detection']['multi_face']['alert_threshold']
        self.consecutive_frames = 0
        self.alert_logger = None

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def detect_multiple_faces(self, frame):
        """
        Returns:
            multiple_faces (bool)
            num_faces (int)
            face_boxes (list of [x1, y1, x2, y2])
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = self.detector.detect(rgb_frame)

        face_boxes = []

        if boxes is not None and len(boxes) > 0:
            # keep only high-confidence faces
            for box, p in zip(boxes, probs):
                if p is None or p < 0.9:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                face_boxes.append([x1, y1, x2, y2])

            num_faces = len(face_boxes)
            multiple_faces = num_faces >= 2

            if multiple_faces:
                self.consecutive_frames += 1
                if self.consecutive_frames >= self.threshold and self.alert_logger:
                    self.alert_logger.log_alert(
                        "MULTIPLE_FACES",
                        f"Detected {num_faces} faces for {self.consecutive_frames} frames"
                    )
            else:
                self.consecutive_frames = 0

        else:
            num_faces = 0
            multiple_faces = False
            self.consecutive_frames = 0

        return multiple_faces, num_faces, face_boxes