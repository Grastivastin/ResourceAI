"""Stage 2 - YOLOv8 identifies material/objects.
   Stage 3 - YOLOv8-seg segments visible cracks / damaged regions.

Two custom weight files are expected in backend/weights/:
  materials.pt    YOLOv8 detect model trained on your material classes
  cracks-seg.pt   YOLOv8-seg model trained on a crack / damage dataset
If they are missing the service runs in DEMO MODE (material = "unknown" and an
OpenCV crack heuristic replaces YOLOv8-seg) so the whole app still works."""
from pathlib import Path
import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


class Detector:
    def __init__(self):
        mat, seg = WEIGHTS / "materials.pt", WEIGHTS / "cracks-seg.pt"
        self.material_model = self.damage_model = None
        if mat.exists():
            from ultralytics import YOLO          # imported lazily: demo mode needs no PyTorch
            self.material_model = YOLO(str(mat))
        if seg.exists():
            from ultralytics import YOLO
            self.damage_model = YOLO(str(seg))
        self.demo_mode = self.material_model is None or self.damage_model is None

    # ----- Stage 2 -----
    def detect_materials(self, img, conf=0.25):
        if self.material_model is None:
            return []
        r = self.material_model.predict(img, conf=conf, verbose=False)[0]
        found = [{"label": r.names[int(b.cls)], "confidence": round(float(b.conf), 3),
                  "box": [round(float(v), 1) for v in b.xyxy[0]]} for b in r.boxes]
        return sorted(found, key=lambda d: -d["confidence"])

    # ----- Stage 3 -----
    def segment_damage(self, img, conf=0.20):
        """Return a binary mask (1 = damaged pixel) the same size as img."""
        h, w = img.shape[:2]
        if self.damage_model is None:
            return self._classical_cracks(img)
        mask = np.zeros((h, w), np.uint8)
        r = self.damage_model.predict(img, conf=conf, verbose=False)[0]
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask |= (m > 0.5).astype(np.uint8)
        return mask

    @staticmethod
    def _classical_cracks(img, min_len=40, min_slender=6.0, k=3.0):
        """Fallback crack finder (used only when cracks-seg.pt is missing).
        1) black-hat keeps thin DARK structures  2) adaptive threshold ignores normal texture
        3) keep only LONG + SLENDER components (length / thickness): cracks wander but stay
           thin, while pits and stains are compact blobs."""
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        thr = max(20.0, float(bh.mean() + k * bh.std()))            # relative to this image's texture
        binary = (bh > thr).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        keep = np.zeros_like(binary)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            length = float(np.hypot(w, h))                          # bounding-box diagonal
            if area < 30 or length < min_len:
                continue
            thickness = max(2.0 * dist[labels == i].max(), 1.0)
            if length / thickness >= min_slender:
                keep[labels == i] = 1
        return keep
