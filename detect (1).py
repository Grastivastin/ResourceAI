"""Stage 2 - YOLOv8 identifies material/objects.
   Stage 3 - YOLOv8-seg segments visible cracks / damaged regions.
   Stage 2.5/5 - Claude LLM: vision fallback/cross-check + natural-language explanation.

Two custom weight files are expected in backend/weights/:
  materials.pt    YOLOv8 detect model trained on your material classes
  cracks-seg.pt   YOLOv8-seg model trained on a crack / damage dataset
If they are missing the service runs in DEMO MODE (material = "unknown" and an
OpenCV crack heuristic replaces YOLOv8-seg) so the whole app still works.

Claude integration (new): set ANTHROPIC_API_KEY to enable it. It never replaces
the trained CV/ML models -- it does two honest, separate jobs:
  1. claude_assess()      a vision cross-check / fallback material+damage read,
                           used when YOLO weights aren't present yet.
  2. generate_explanation() turns the ALREADY-DECIDED structured recommendation
                           (from pipeline.recommend) into a human-readable
                           paragraph. It explains the decision, it doesn't make it.
Both degrade to None/a template string with no API key configured, matching the
existing demo_mode pattern below -- the app never breaks if the key is missing.
"""
from pathlib import Path
import base64
import json
import os
import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"

# Swap to "claude-sonnet-5" for higher-quality output if latency isn't a concern
# in your demo; haiku is the fast/cheap choice for a live, on-stage scan.
CLAUDE_VISION_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_TEXT_MODEL = "claude-haiku-4-5-20251001"


def _claude_client():
    """Lazily build an Anthropic client. Returns None (never raises) if no key
    is configured, so every caller can just check for None and fall back --
    exactly like self.material_model is None above."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        return None


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
        self.claude = _claude_client()

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

    # ----- Stage 2.5: Claude vision fallback / cross-check -----
    def claude_assess(self, img):
        """Ask Claude to independently read material + visible damage straight from
        the pixels. Used two ways: (a) as the detector when demo_mode is True and no
        trained YOLO weights exist yet, so the pipeline still gives a real answer
        instead of 'unknown'; (b) as an optional cross-check alongside YOLO even once
        weights exist, surfaced in the report as a second opinion, not a silent override.
        Returns None if no ANTHROPIC_API_KEY is set (caller must handle that, same as
        checking self.material_model is None above). Returns a dict with an "error"
        key, rather than raising, if the call itself fails (bad key, network, etc.)
        so one flaky API call never crashes a live scan."""
        if self.claude is None:
            return None
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
        prompt = (
            "You are a cross-check step in a construction-demolition material recovery "
            "pipeline. Identify the single dominant material in this image from exactly "
            "this list: brick, concrete, tile, wood, gypsum, plastic, metal, unknown. "
            "Then describe visible condition/damage in one short sentence (cracks, "
            "fragmentation, staining, or 'no visible damage'), and give a rough visible-"
            "damage severity from 0.0 (pristine) to 1.0 (severely damaged) based ONLY on "
            "what you can see, not on guessing structural safety. "
            'Respond with ONLY compact JSON, no prose: '
            '{"material": "...", "damage_description": "...", "damage_severity": 0.0}'
        )
        try:
            resp = self.claude.messages.create(
                model=CLAUDE_VISION_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            text = resp.content[0].text.strip()
            text = text[text.find("{"): text.rfind("}") + 1]        # strip any stray preamble
            parsed = json.loads(text)
            parsed["source"] = "claude-vision"
            return parsed
        except Exception as e:
            return {"error": str(e), "source": "claude-vision"}


# ----- Stage 5: Claude explains the ALREADY-DECIDED recommendation -----
def generate_explanation(material, condition_score, recommendation, claude_note=None):
    """Turn the structured output of pipeline.recommend.recommend() into a short,
    plain-language paragraph for the report. This function never decides the
    pathway -- `recommendation` is passed in already-computed, exactly as
    pipeline.recommend returns it, so the Random Forest + rule engine remain the
    sole source of the decision. Claude's only job here is to narrate it well.

    Falls back to a template string (no API call) if no key is configured, so the
    report always has *some* explanation text -- this must never block the demo."""
    client = _claude_client()
    pathway = recommendation.get("pathway", "ASSESS")
    uses = ", ".join(recommendation.get("alternative_uses", [])) or "general recovery"
    if client is None:
        return (f"{material.capitalize()} scored {condition_score}/100 on visible "
                f"condition. Recommended pathway: {pathway}. Possible secondary "
                f"uses: {uses}.")
    prompt = (
        f"Write a 2-3 sentence plain-English explanation for a construction site "
        f"report. Facts (do not contradict or re-decide any of these): material = "
        f"{material}, visible-condition score = {condition_score}/100, decided "
        f"pathway = {pathway}, possible secondary uses = {uses}."
        + (f" An independent vision cross-check noted: {claude_note}." if claude_note else "")
        + " Be concrete and avoid hedging filler like 'it is important to note'."
    )
    try:
        resp = client.messages.create(
            model=CLAUDE_TEXT_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return (f"{material.capitalize()} scored {condition_score}/100 on visible "
                f"condition. Recommended pathway: {pathway}. Possible secondary "
                f"uses: {uses}.")
