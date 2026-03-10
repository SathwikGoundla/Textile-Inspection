"""
Textile Object Detection & Classification Module
Uses YOLOv8 for real-time object detection with textile classification gate
"""

import numpy as np
import cv2
import random
import time
from typing import Dict, Any, List

# COCO class labels - IDs that correspond to textile/clothing items
TEXTILE_COCO_CLASSES = {
    "person",           # Often wearing textiles
    "tie",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    # clothing items
    "shirt", "cloth", "fabric", "textile", "garment",
    "jacket", "pants", "dress", "skirt", "shorts",
    "sweater", "hoodie", "coat", "jeans", "t-shirt",
    "blouse", "curtain", "bed", "pillow",
}

# Textile type taxonomy
TEXTILE_TYPES = [
    "Cotton Fabric", "Denim", "Silk", "Polyester", "Wool",
    "Linen", "Nylon Mesh", "Knit Fabric", "Woven Cloth",
    "Synthetic Blend", "Canvas", "Fleece"
]


class TextileDetector:
    """
    Handles object detection and textile classification.
    In production: loads actual YOLOv8 model.
    In demo/test: uses sophisticated simulation.
    """

    def __init__(self, model_path: str = None, use_gpu: bool = False):
        self.model = None
        self.model_loaded = False
        self.use_simulation = True
        self._frame_counter = 0
        self._scenario_cycle = 0
        self._scenarios = self._build_scenarios()

        # Try loading actual YOLO model
        try:
            from ultralytics import YOLO
            model_file = model_path or "yolov8n.pt"
            self.model = YOLO(model_file)
            self.model_loaded = True
            self.use_simulation = False
            print(f"✅ YOLOv8 model loaded: {model_file}")
        except ImportError:
            print("⚠️  ultralytics not installed - using simulation mode")
        except Exception as e:
            print(f"⚠️  Model load failed ({e}) - using simulation mode")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main detection pipeline.
        Returns: {is_textile, textile_type, confidence, boxes}
        """
        self._frame_counter += 1

        if not self.use_simulation and self.model_loaded:
            return self._detect_real(frame)
        else:
            return self._detect_simulated(frame)

    def _detect_real(self, frame: np.ndarray) -> Dict[str, Any]:
        """Use actual YOLOv8 model for detection"""
        results = self.model(frame, verbose=False, conf=0.4)[0]
        boxes_out = []
        is_textile = False
        best_conf = 0.0
        textile_type = None

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id].lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxyn[0].tolist()

            boxes_out.append({
                "label": label,
                "conf": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

            # Textile gate classification
            if self._is_textile_class(label, conf):
                if conf > best_conf:
                    best_conf = conf
                    textile_type = self._map_textile_type(label, frame)
                    is_textile = True

        return {
            "is_textile": is_textile,
            "textile_type": textile_type,
            "confidence": round(best_conf, 3),
            "boxes": boxes_out
        }

    def _detect_simulated(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Sophisticated simulation for demo/testing.
        Cycles through realistic detection scenarios every ~90 frames.
        Uses actual frame content analysis for realism.
        """
        # Change scenario every ~3 seconds at 30fps
        if self._frame_counter % 90 == 1:
            self._scenario_cycle = (self._scenario_cycle + 1) % len(self._scenarios)

        scenario = self._scenarios[self._scenario_cycle]

        # Add slight confidence jitter for realism
        jitter = (random.random() - 0.5) * 0.04
        conf = max(0.0, min(1.0, scenario["confidence"] + jitter))

        # Analyze actual frame brightness/texture for dynamic response
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        edge_density = np.mean(cv2.Canny(gray, 50, 150)) / 255.0
        has_content = brightness > 0.05 and edge_density > 0.01

        if not has_content and scenario["is_textile"]:
            # Very dark/uniform frame — lower confidence
            conf *= 0.7

        return {
            "is_textile": scenario["is_textile"] and has_content,
            "textile_type": scenario["textile_type"] if scenario["is_textile"] else None,
            "confidence": round(conf, 3),
            "boxes": scenario["boxes"] if has_content else []
        }

    def _is_textile_class(self, label: str, conf: float) -> bool:
        """Multi-strategy textile classification"""
        if conf < 0.35:
            return False
        # Direct match
        for t in TEXTILE_COCO_CLASSES:
            if t in label or label in t:
                return True
        # Keyword-based
        textile_keywords = [
            "fabric", "cloth", "textile", "cotton", "woven",
            "knit", "denim", "wool", "silk", "polyester",
            "garment", "apparel", "shirt", "pants", "dress"
        ]
        return any(kw in label for kw in textile_keywords)

    def _map_textile_type(self, label: str, frame: np.ndarray) -> str:
        """Map detected label + visual features to textile type"""
        label_mapping = {
            "denim": "Denim", "jeans": "Denim",
            "silk": "Silk", "satin": "Silk",
            "cotton": "Cotton Fabric", "shirt": "Cotton Fabric",
            "wool": "Wool", "sweater": "Wool",
            "mesh": "Nylon Mesh", "net": "Nylon Mesh",
            "canvas": "Canvas", "linen": "Linen",
        }
        for key, ttype in label_mapping.items():
            if key in label:
                return ttype

        # Use color analysis to guess fabric type
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:,:,0])
        avg_sat = np.mean(hsv[:,:,1])
        if avg_hue < 15 or avg_hue > 170:  # Red-ish
            return "Polyester Blend"
        elif 90 < avg_hue < 140 and avg_sat > 60:  # Blue
            return "Denim"
        elif avg_sat < 30:  # Desaturated
            return "Cotton Fabric"
        return random.choice(TEXTILE_TYPES)

    def _build_scenarios(self) -> List[Dict]:
        """Pre-built realistic detection scenarios"""
        return [
            {
                "is_textile": True,
                "textile_type": "Cotton Fabric",
                "confidence": 0.91,
                "boxes": [{"label": "fabric", "conf": 0.91, "x1": 0.1, "y1": 0.15, "x2": 0.9, "y2": 0.85}]
            },
            {
                "is_textile": True,
                "textile_type": "Denim",
                "confidence": 0.87,
                "boxes": [{"label": "denim", "conf": 0.87, "x1": 0.05, "y1": 0.1, "x2": 0.95, "y2": 0.9}]
            },
            {
                "is_textile": True,
                "textile_type": "Polyester Blend",
                "confidence": 0.82,
                "boxes": [{"label": "fabric", "conf": 0.82, "x1": 0.15, "y1": 0.2, "x2": 0.85, "y2": 0.8}]
            },
            {
                "is_textile": False,  # Non-textile object
                "textile_type": None,
                "confidence": 0.0,
                "boxes": [{"label": "bottle", "conf": 0.78, "x1": 0.3, "y1": 0.2, "x2": 0.7, "y2": 0.8}]
            },
            {
                "is_textile": True,
                "textile_type": "Silk",
                "confidence": 0.94,
                "boxes": [{"label": "silk cloth", "conf": 0.94, "x1": 0.08, "y1": 0.12, "x2": 0.92, "y2": 0.88}]
            },
            {
                "is_textile": True,
                "textile_type": "Wool",
                "confidence": 0.79,
                "boxes": [{"label": "wool fabric", "conf": 0.79, "x1": 0.12, "y1": 0.18, "x2": 0.88, "y2": 0.82}]
            },
            {
                "is_textile": False,  # No object
                "textile_type": None,
                "confidence": 0.0,
                "boxes": []
            },
        ]
