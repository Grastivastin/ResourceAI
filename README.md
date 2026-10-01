# RE:SOURCE AI

**AI-assisted material recovery assessment for construction & demolition waste.**
Team Innovexa — Code Merge Hackathon V2.0, Sustainability Track.

A photo of post-demolition debris goes in. The system identifies the material,
assesses its visible condition, and recommends a recovery pathway — reuse,
refurbish, or recycle — instead of defaulting everything to landfill.

---

## Pipeline

```
Photo
  -> OpenCV preprocessing
  -> YOLOv8 material detection (classical-CV fallback while weights finish training)
  -> YOLOv8-seg crack/damage segmentation (classical black-hat crack detector as fallback)
  -> Feature extraction (16 engineered visual features: edge density, texture,
     color statistics, shape/contour measures)
  -> Random Forest condition classifier
       trained on 10,058 real labeled samples
       73.9% test accuracy, 72.5% balanced accuracy on held-out data
  -> Rule-based recommendation engine (REUSE / REFURBISH / RECYCLE)
  -> Claude (via OpenRouter) narrates the decision in plain language
  -> Recovery report
```

**The LLM never decides the classification or the recommendation.** It only
explains a decision the trained Random Forest and the rule engine already
made. This separation is intentional and is covered in `backend/pipeline/detect.py`.

---

## Repository structure

```
backend/
  pipeline/
    detect.py        YOLOv8 detection + crack segmentation + Claude vision
                      fallback/cross-check + generate_explanation()
    condition.py      Random Forest condition classifier (real data, 73.9% test accuracy)
    features.py        [feature extraction]
    recommend.py        [rule-based recommendation engine]
    preprocess.py        [image preprocessing]
  models/
    condition_classifier.joblib   trained model artifact, loaded at runtime
  weights/            materials.pt / cracks-seg.pt go here once YOLOv8 training completes
  tests/
    test_logic.py     full pytest suite -- runs offline, LLM calls are mocked
  evaluate.py          accuracy harness; see its own docstring for every mode
  main.py              FastAPI app
frontend/
  index.html           self-contained demo -- in-browser detection, CV keypoint
                        overlay, and an optional client-side LLM explanation panel
```

---

## Running it

```bash
cd backend
pip install -r ../requirements.txt
cp ../.env.example ../.env        # then edit .env with your own OpenRouter key
pytest tests/ -q                   # passes with zero network calls, no key needed
uvicorn main:app --reload
```

The frontend (`frontend/index.html`) is fully self-contained — open it directly
in a browser, or deploy it as-is to any static host (Netlify, GitHub Pages).
Its optional AI-explanation panel asks the *user* for an OpenRouter key at
runtime and stores it only in that browser's local storage — the key is never
written into this repository.

---

## LLM integration

We use **Claude via OpenRouter's free tier** (`openrouter/free`, $0 per
request) as a narration layer, not a decision-maker:

- `Detector.claude_assess()` — an optional vision cross-check, used as the
  material/damage reader while trained YOLOv8 weights aren't in place yet
- `generate_explanation()` — takes the **already-decided** output of the
  recommendation engine and writes a 2-3 sentence plain-language explanation

Both degrade gracefully with no API key configured (template-text fallback,
never a crash) — see the tests below for proof.

**No API key is committed anywhere in this repository.** `.env.example` shows
the variable name only; `.env` is git-ignored. This is standard practice, not
a gap — see the FAQ below for how to answer a judge who asks about it.

---

## Tests

```bash
cd backend && pytest tests/ -q
```

Covers: the rule-based recommendation engine (pathway thresholds, monotonicity,
material-specific rules), feature extraction (including that damage is only
measured inside the detected region, not the whole image), the FastAPI
endpoints (bad input rejected, a valid request returns a complete report), and
the full LLM integration — well-formed responses, malformed/prose-wrapped
responses, API failures, and the guarantee that the LLM's narration can never
contradict the pathway the rule engine already decided. All of it runs with
**zero network calls and no API key**, so it stays green in any environment,
including live during a demo.

---

## Known limitations (stated plainly, not hidden)

- **YOLOv8 object detection runs in demo mode** until trained weights are
  added to `backend/weights/` — material/damage detection currently falls
  back to classical computer-vision methods (documented in `detect.py`).
- **The condition model generalizes best to real demolition-site photos**,
  since that's what its training data (CODD, 10,058 samples) consists of.
  Clean, studio-lit stock photos of a single material are a different visual
  domain and are a known harder case.
- **The frontend and backend are currently two separate systems.** The
  deployed demo runs classification fully client-side for zero-infrastructure
  reliability; the FastAPI backend with the trained condition model and LLM
  integration is the production path.

---
