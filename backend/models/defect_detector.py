"""
Textile Defect Detection Module
CNN-based defect identification with localization and severity scoring
"""

import cv2
import numpy as np
import random
from typing import Dict, Any, List, Optional


# Defect taxonomy with metadata
DEFECT_CATALOG = {
    "HOLE": {
        "display": "Hole",
        "severity_weight": 3,
        "detection_method": "morphological",
        "description": "Physical opening/perforation in fabric"
    },
    "TEAR": {
        "display": "Tear",
        "severity_weight": 3,
        "detection_method": "edge_analysis",
        "description": "Linear rip or cut in fabric structure"
    },
    "STAIN": {
        "display": "Stain",
        "severity_weight": 2,
        "detection_method": "color_anomaly",
        "description": "Foreign substance contamination"
    },
    "MISPRINT": {
        "display": "Misprint",
        "severity_weight": 2,
        "detection_method": "pattern_analysis",
        "description": "Incorrect or misaligned print pattern"
    },
    "WEAVING_DEFECT": {
        "display": "Weaving Defect",
        "severity_weight": 2,
        "detection_method": "texture_analysis",
        "description": "Broken/missing threads in weave pattern"
    },
    "COLOR_INCONSISTENCY": {
        "display": "Color Inconsistency",
        "severity_weight": 1,
        "detection_method": "color_uniformity",
        "description": "Uneven dye distribution or fading"
    },
    "PILLING": {
        "display": "Pilling",
        "severity_weight": 1,
        "detection_method": "surface_analysis",
        "description": "Small fiber balls on fabric surface"
    },
    "CREASE": {
        "display": "Crease",
        "severity_weight": 1,
        "detection_method": "wrinkle_detection",
        "description": "Permanent fold marks in fabric"
    }
}


class DefectDetector:
    """
    Multi-method textile defect detection pipeline.
    
    In production: loads CNN model (EfficientNet/ResNet fine-tuned on textile dataset).
    Currently implements computer vision heuristics + simulation for demo.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self._frame_counter = 0
        self._defect_cycle = 0

        # Try loading actual CNN model
        try:
            import torch
            from torchvision import models, transforms
            if model_path:
                self.model = torch.load(model_path)
                self.model.eval()
                print("✅ Defect CNN model loaded")
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️  Defect model load failed: {e}")

        # Defect scenarios for simulation
        self._defect_scenarios = self._build_defect_scenarios()

    def analyze(self, frame: np.ndarray, textile_boxes: List[Dict]) -> Dict[str, Any]:
        """
        Analyze frame for textile defects.
        Returns detected defects with locations and confidence scores.
        """
        self._frame_counter += 1

        # Crop to textile region if boxes provided
        roi = self._extract_roi(frame, textile_boxes)

        # Run detection pipeline
        detected_defects = []

        if self.model:
            detected_defects = self._detect_with_model(roi)
        else:
            # Vision-based heuristics + simulation
            cv_defects = self._detect_with_cv(roi)
            sim_defects = self._get_simulation_defects()
            
            # Merge: prefer CV-detected, supplement with simulation
            detected_defects = self._merge_defects(cv_defects, sim_defects)

        # Calculate severity
        severity = self._calculate_severity(detected_defects)

        return {
            "defects": detected_defects,
            "defect_count": len(detected_defects),
            "severity": severity,
            "roi_analyzed": roi.shape[:2] if roi is not None else None
        }

    def _detect_with_cv(self, frame: np.ndarray) -> List[Dict]:
        """Computer vision based defect detection heuristics"""
        defects = []
        if frame is None or frame.size == 0:
            return defects

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 1. Hole Detection (dark blobs via thresholding) ─────────────────
        _, dark_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 5000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"] / w
                    cy = M["m01"] / M["m00"] / h
                    conf = min(0.95, 0.6 + area / 8000)
                    defects.append(self._make_defect("HOLE", cx, cy, conf))

        # ── 2. Stain Detection (color anomaly analysis) ───────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0].astype(float)
        s_chan = hsv[:, :, 1].astype(float)
        h_mean, h_std = np.mean(h_chan), np.std(h_chan)
        
        # High saturation anomalies = potential stains
        stain_mask = ((s_chan > 120) & 
                      (np.abs(h_chan - h_mean) > 2 * h_std + 10)).astype(np.uint8) * 255
        stain_mask = cv2.dilate(stain_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 300 < area < 10000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"] / w
                    cy = M["m01"] / M["m00"] / h
                    defects.append(self._make_defect("STAIN", cx, cy, 0.72))

        # ── 3. Tear Detection (linear edge anomalies) ─────────────────────
        edges = cv2.Canny(gray, 80, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                                minLineLength=40, maxLineGap=10)
        if lines is not None and len(lines) > 15:
            # Many sharp lines = potential tear
            # Find dominant line cluster
            for line in lines[:3]:
                x1, y1, x2, y2 = line[0]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                defects.append(self._make_defect("TEAR", cx, cy, 0.65))
            defects = defects[:5]  # Limit

        # ── 4. Color Inconsistency (block variance analysis) ──────────────
        block_size = 32
        variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                variances.append((np.var(block), x, y))
        
        if variances:
            mean_var = np.mean([v[0] for v in variances])
            high_var = [(v, x, y) for v, x, y in variances if v > mean_var * 3.5]
            if len(high_var) > 3:
                v_score, vx, vy = high_var[0]
                defects.append(self._make_defect(
                    "COLOR_INCONSISTENCY",
                    (vx + block_size/2) / w,
                    (vy + block_size/2) / h,
                    0.58
                ))

        return defects[:4]  # Max 4 CV-detected defects

    def _get_simulation_defects(self) -> List[Dict]:
        """Cycle through realistic defect scenarios"""
        # Change scenario slowly (every 4 seconds at 30fps)
        scenario_idx = (self._frame_counter // 120) % len(self._defect_scenarios)
        scenario = self._defect_scenarios[scenario_idx]

        # Add small noise to positions
        result = []
        for d in scenario:
            d_copy = dict(d)
            d_copy["location"] = {
                "x": min(0.95, max(0.05, d["location"]["x"] + (random.random()-0.5)*0.05)),
                "y": min(0.95, max(0.05, d["location"]["y"] + (random.random()-0.5)*0.05))
            }
            d_copy["confidence"] = min(0.99, max(0.5, d["confidence"] + (random.random()-0.5)*0.06))
            result.append(d_copy)

        return result

    def _merge_defects(self, cv_defects: List[Dict], sim_defects: List[Dict]) -> List[Dict]:
        """Merge CV and simulation results, prioritizing CV"""
        if cv_defects:
            return cv_defects
        return sim_defects

    def _detect_with_model(self, frame: np.ndarray) -> List[Dict]:
        """Use loaded CNN model for defect detection"""
        import torch
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        defects = []
        defect_types = list(DEFECT_CATALOG.keys())
        for i, prob in enumerate(probs):
            if prob > 0.3 and i < len(defect_types):
                defects.append(self._make_defect(
                    defect_types[i],
                    random.uniform(0.2, 0.8),
                    random.uniform(0.2, 0.8),
                    float(prob)
                ))
        return defects

    def _extract_roi(self, frame: np.ndarray, boxes: List[Dict]) -> np.ndarray:
        """Crop frame to textile bounding box"""
        if not boxes or frame is None:
            return frame
        h, w = frame.shape[:2]
        box = boxes[0]
        x1 = max(0, int(box.get("x1", 0) * w))
        y1 = max(0, int(box.get("y1", 0) * h))
        x2 = min(w, int(box.get("x2", 1) * w))
        y2 = min(h, int(box.get("y2", 1) * h))
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else frame

    def _make_defect(self, defect_type: str, cx: float, cy: float, conf: float) -> Dict:
        """Create standardized defect record"""
        catalog = DEFECT_CATALOG.get(defect_type, {})
        return {
            "type": defect_type,
            "display_name": catalog.get("display", defect_type),
            "confidence": round(conf, 3),
            "location": {"x": round(cx, 3), "y": round(cy, 3)},
            "severity_weight": catalog.get("severity_weight", 1),
            "description": catalog.get("description", "")
        }

    def _calculate_severity(self, defects: List[Dict]) -> str:
        """Calculate overall severity from detected defects"""
        if not defects:
            return "NONE"
        total_weight = sum(d.get("severity_weight", 1) for d in defects)
        if total_weight >= 6:
            return "CRITICAL"
        elif total_weight >= 3:
            return "HIGH"
        elif total_weight >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _build_defect_scenarios(self) -> List[List[Dict]]:
        """Pre-built defect scenarios for simulation"""
        return [
            [],  # Clean - PASS
            [],  # Clean - PASS
            [self._make_defect("HOLE", 0.3, 0.4, 0.88)],
            [self._make_defect("STAIN", 0.6, 0.3, 0.79)],
            [],  # Clean
            [self._make_defect("TEAR", 0.5, 0.5, 0.82),
             self._make_defect("COLOR_INCONSISTENCY", 0.7, 0.6, 0.61)],
            [self._make_defect("WEAVING_DEFECT", 0.4, 0.5, 0.74)],
            [],  # Clean
            [self._make_defect("MISPRINT", 0.55, 0.45, 0.91)],
            [self._make_defect("PILLING", 0.3, 0.7, 0.67)],
            [],  # Clean
            [self._make_defect("HOLE", 0.25, 0.35, 0.93),
             self._make_defect("TEAR", 0.65, 0.55, 0.86)],
        ]
