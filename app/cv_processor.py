# app/cv_processor.py

import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import base64

from ultralytics import YOLO
import torch

MODEL_PATH = r"C:\Users\konom\exam-monitoring\models\Detection_with_hands_v3_HDYOLO11.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BehaviorDetector:
    """
    Wrapper for your PT algorithm (now YOLO).
    """

    def __init__(self):
        print("Loading YOLO model for behavior detection...")
        self.model = YOLO(MODEL_PATH).to(DEVICE)  # load model on GPU if available[web:88]
        self.initialized = True
        print(f"Model loaded on device: {DEVICE}")

    def decode_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 image to OpenCV format"""
        try:
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")

    def detect_behaviors(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO on the frame and convert detections into your behavior format.
        """
        behaviors: List[Dict] = []

        # Run YOLO
        results = self.model.predict(frame, device=DEVICE)[0]  # one Results object[web:85]

        boxes_xyxy = results.boxes.xyxy.tolist()   # [[x1,y1,x2,y2], ...][web:156]
        confs = results.boxes.conf.tolist()
        class_ids = results.boxes.cls.tolist()

        for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, class_ids):
            cls_id = int(cls_id)
            label = self.model.names[cls_id] if hasattr(self.model, "names") else str(cls_id)

            # Map model class → behavior label/severity (customize as needed)
            behavior_label = label  # e.g. "hand", "phone", etc.
            severity = "medium"
            if "phone" in label.lower():
                severity = "critical"
            elif "hand" in label.lower():
                severity = "high"

            bbox = {
                "x": float(x1),
                "y": float(y1),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            }

            behaviors.append({
                "behavior_label": behavior_label,
                "confidence": float(conf),
                "severity": severity,
                "bbox": bbox,
                "extra_data": {
                    "class_id": cls_id,
                    "class_name": label,
                },
            })

        # Optionally, keep your old rules (e.g., "no_face_detected") if there are no detections

        return behaviors

    def process_frame(self, base64_image: str, camera_id: int) -> Dict:
        """Process a single frame and return detected behaviors"""
        try:
            frame = self.decode_image(base64_image)
            if frame is None:
                return {"success": False, "error": "Failed to decode image"}

            behaviors = self.detect_behaviors(frame)
            timestamp = datetime.now().isoformat()

            return {
                "success": True,
                "camera_id": camera_id,
                "timestamp": timestamp,
                "frame_shape": frame.shape,
                "behaviors": behaviors,
                "behavior_count": len(behaviors),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global detector instance used by app.main
detector = BehaviorDetector()
