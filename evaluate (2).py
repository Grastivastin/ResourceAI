"""Accuracy harness for every stage.

  python evaluate.py damage [--seed 99] [--hard]     synthetic self-test of the crack stage
  python evaluate.py damage --images I/ --masks M/   REAL test: photos + ground-truth masks
  python evaluate.py condition [data.csv]            Random Forest cross-validation
  python evaluate.py pathway data.csv                pathway agreement (needs 'pathway' column)
  python evaluate.py yolo materials.yaml cracks.yaml YOLOv8 / YOLOv8-seg mAP (needs weights)
  python evaluate.py claude --images I/ [--labels L.csv]   Claude vision smoke test / agreement
  python evaluate.py explain                         prints a sample Claude explanation, no images needed
"""
import sys
from pathlib import Path
import cv2
import numpy as np

from pipeline.preprocess import preprocess
from pipeline.detect import Detector, WEIGHTS, generate_explanation


# ------------------------------------------------------------------ crack stage
def tolerant_scores(pred, gt, tol=3):
    """Crack lines are 2-4 px wide, so a 1-px shift is not an error: a predicted pixel
    counts as correct if it lies within `tol` px of a true crack pixel (and vice versa)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gt_d, pr_d = cv2.dilate(gt, k), cv2.dilate(pred, k)
    prec = (pred & gt_d).sum() / max(pred.sum(), 1)
    rec = (gt & pr_d).sum() / max(gt.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = (pred & gt).sum() / max((pred | gt).sum(), 1)            # strict pixel IoU
    return prec, rec, f1, iou


def synthetic_surface(rng, cracks=True, h=480, w=640, hard=False):
    """Textured grey surface (concrete-like) + lighting gradient + dark pits as decoys."""
    base = rng.normal(0, 14, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.2) * 2.2
    grad = np.linspace(-25, 25, w)[None, :] + np.linspace(-15, 15, h)[:, None]
    img = np.clip(145 + base + grad, 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for _ in range(rng.integers(8, 20)):                           # pits / stains (NOT cracks)
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(2, 6)), int(rng.integers(70, 110)), -1)
    if cracks:
        for _ in range(rng.integers(1, 4)):
            p = np.array([rng.integers(0, w), rng.integers(0, h)], float)
            ang = rng.uniform(0, 2 * np.pi)
            for _ in range(rng.integers(12, 30)):
                ang += rng.normal(0, 0.35)
                q = p + 18 * np.array([np.cos(ang), np.sin(ang)])
                t = int(rng.integers(1, 3)) if hard else int(rng.integers(2, 5))
                cv2.line(mask, tuple(p.astype(int)), tuple(q.astype(int)), 1, t)
                cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), int(rng.integers(100, 125)) if hard else int(rng.integers(35, 70)), t)
                p = q
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), mask


def eval_damage(images=None, masks=None, n=60, seed=11, hard=False):
    det = Detector()
    rows, img_hits = [], []
    if images:                                                     # REAL data
        pairs = [(cv2.imread(str(p)), (cv2.imread(str(Path(masks) / (p.stem + ".png")), 0) > 127).astype(np.uint8))
                 for p in sorted(Path(images).glob("*.*"))]
        label = f"REAL data ({len(pairs)} images, mode={'demo' if det.demo_mode else 'YOLOv8-seg'})"
    else:                                                          # synthetic self-test
        rng = np.random.default_rng(seed)
        pairs = [synthetic_surface(rng, cracks=(i % 5 != 0), hard=hard) for i in range(n)]
        label = f"SYNTHETIC {'HARD (faint hairline cracks)' if hard else 'standard'} self-test, seed {seed} ({n} images, 20% crack-free) - NOT real-world accuracy"
    for img, gt in pairs:
        img = preprocess(img)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred = det.segment_damage(img)
        has_gt, has_pred = gt.sum() > 0, pred.sum() / pred.size > 0.002
        img_hits.append(has_gt == has_pred)
        if has_gt:
            rows.append(tolerant_scores(pred, gt))
    r = np.mean(rows, axis=0)
    print(label)
    print(f"  precision {r[0]:.2f} | recall {r[1]:.2f} | F1 {r[2]:.2f} | strict IoU {r[3]:.2f}")
    print(f"  image-level 'cracked vs sound' accuracy: {np.mean(img_hits):.2%}")


# ------------------------------------------------------------------ condition model
def eval_condition(csv=None):
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from pipeline.condition import synthetic_dataset
    from pipeline.features import FEATURE_NAMES
    if csv:
        import pandas as pd
        df = pd.read_csv(csv); X, y = df[FEATURE_NAMES], df["condition_score"]; tag = f"REAL data {csv}"
    else:
        X, y = synthetic_dataset(); tag = "SYNTHETIC placeholder data - only proves the code works"
    pred = cross_val_predict(RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=7),
                             X, y, cv=KFold(5, shuffle=True, random_state=7))
    grade = lambda s: np.where(s >= 70, "Good", np.where(s >= 40, "Fair", "Poor"))
    print(f"Random Forest, 5-fold CV on {tag}")
    print(f"  MAE {mean_absolute_error(y, pred):.1f} points | R^2 {r2_score(y, pred):.3f} "
          f"| grade accuracy {np.mean(grade(y) == grade(pred)):.2%}")


# ------------------------------------------------------------------ pathway agreement
def eval_pathway(csv):
    import pandas as pd
    from sklearn.metrics import classification_report
    from pipeline.recommend import recommend
    df = pd.read_csv(csv)
    pred = [recommend(r.material, r.condition_score, r._asdict())["pathway"] for r in df.itertuples()]
    print(classification_report(df["pathway"].str.upper(), pred, zero_division=0))


# ------------------------------------------------------------------ YOLO
def eval_yolo(mat_yaml, seg_yaml):
    from ultralytics import YOLO
    m = YOLO(str(WEIGHTS / "materials.pt")).val(data=mat_yaml)
    print(f"Material detection  mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f}")
    s = YOLO(str(WEIGHTS / "cracks-seg.pt")).val(data=seg_yaml)
    print(f"Crack segmentation  mask mAP50 {s.seg.map50:.3f} | mask mAP50-95 {s.seg.map:.3f}")


# ------------------------------------------------------------------ Claude vision
def eval_claude(images, labels=None):
    """Smoke test / agreement check for the Claude vision fallback. With --labels
    (a CSV with columns filename,material) it reports accuracy against ground truth,
    same spirit as the other eval modes. Without it, just prints what Claude sees,
    so you can sanity-check the integration is wired and the key works before a demo."""
    det = Detector()
    if det.claude is None:
        print("ANTHROPIC_API_KEY not set -- nothing to evaluate. Export it and retry.")
        return
    gt = {}
    if labels:
        import pandas as pd
        df = pd.read_csv(labels)
        gt = dict(zip(df["filename"], df["material"]))
    hits, total = 0, 0
    for p in sorted(Path(images).glob("*.*")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = det.claude_assess(img)
        label = result.get("material", "ERROR") if result else "NO-KEY"
        sev = result.get("damage_severity") if result else None
        if p.name in gt:
            total += 1
            ok = label == gt[p.name]
            hits += ok
            print(f"  {p.name}: claude={label} ({'OK' if ok else 'expected ' + gt[p.name]}) severity={sev}")
        else:
            print(f"  {p.name}: claude={label} severity={sev} -- {result.get('damage_description','') if result else ''}")
    if total:
        print(f"\nClaude vision agreement: {hits}/{total} = {hits/total:.2%}")


def eval_explain():
    """No images needed -- just proves the explanation call works end-to-end with a
    canned recommendation, so you can sanity-check API connectivity in 2 seconds."""
    sample_rec = {"pathway": "REFURBISH", "alternative_uses": ["landscaping blocks", "non-structural partition"]}
    text = generate_explanation("brick", 62, sample_rec, claude_note="minor surface cracking visible")
    print("Sample Claude explanation:\n")
    print(" ", text)


if __name__ == "__main__":
    a = sys.argv[1:] or ["damage"]
    opt = lambda f: a[a.index(f) + 1] if f in a else None
    {"damage": lambda: eval_damage(opt("--images"), opt("--masks"), seed=int(opt("--seed") or 11), hard="--hard" in a),
     "condition": lambda: eval_condition(a[1] if len(a) > 1 else None),
     "pathway": lambda: eval_pathway(a[1]),
     "yolo": lambda: eval_yolo(a[1], a[2]),
     "claude": lambda: eval_claude(opt("--images"), opt("--labels")),
     "explain": lambda: eval_explain()}[a[0]]()
