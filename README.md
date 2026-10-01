# RE:SOURCE AI

AI-assisted material recovery assessment for construction & demolition waste.

## Pipeline

```
Photo -> OpenCV preprocessing -> YOLOv8 material detection
      -> YOLOv8-seg crack/damage segmentation (classical-CV fallback if no trained weights yet)
      -> Feature extraction (16 engineered features)
      -> Random Forest condition classifier (trained on 10,058 real labeled samples,
         73.9% held-out test accuracy)
      -> Rule-based recommendation engine (REUSE / REFURBISH / RECYCLE)
      -> Claude (via OpenRouter) narrates the decision in plain language
      -> Recovery report
```

The LLM (Claude, via OpenRouter's free tier) **never decides the classification or
recommendation** -- it only explains a decision the trained models already made.
See `backend/pipeline/detect.py` docstrings for exactly where it sits.

## Setup

```bash
cd backend
pip install -r ../requirements.txt
cp ../.env.example ../.env   # then edit .env with your real OpenRouter key
pytest tests/ -q              # all tests run offline, no API key required
```

## Project structure

```
backend/
  pipeline/
    detect.py       YOLOv8 detection + crack segmentation + Claude vision fallback/explanation
    condition.py     Random Forest condition classifier (real data, 73.9% test accuracy)
    features.py      [existing -- not modified]
    recommend.py      [existing -- not modified]
    preprocess.py     [existing -- not modified]
  models/
    condition_classifier.joblib   trained model artifact
  weights/            materials.pt / cracks-seg.pt go here once trained (see evaluate.py)
  tests/
    test_logic.py     full suite, Claude calls mocked -- no network needed to pass
  evaluate.py          accuracy harness for every stage, see its docstring for all modes
  main.py              [existing -- not modified, needs one new line, see below]
frontend/
  index.html           self-contained demo UI
docs/
  (architecture notes, pitch materials)
```

## Wiring the LLM into main.py

In your `/api/assess` handler, after `recommend(...)` is called:

```python
from pipeline.detect import generate_explanation
explanation = generate_explanation(material, condition_score, recommendation)
# add `explanation` to the returned report dict
```
