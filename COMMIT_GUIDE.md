# RE:SOURCE AI — Step-by-step GitHub commit guide

18 commits, one per feature. Every commit has been checked: tests pass at each one, and nothing is left uncommitted at the end.

## 0. One-time setup

1. Unzip `resource-ai.zip`. You get a `resource-ai/` folder.
2. On GitHub, create a **new empty repository** called `resource-ai` (no README, no .gitignore, no license — the project already has them).
3. Open a terminal **inside** the `resource-ai/` folder:

```bash
cd resource-ai
git init -b main
git config user.name  "Your Name"          # skip if already set globally
git config user.email "you@example.com"
git remote add origin https://github.com/<your-username>/resource-ai.git
```

Optional but recommended — check it runs before committing:

```bash
npm test                      # 16 tests, needs Node 18+
python -m http.server 8080    # then open http://localhost:8080
```

> If you already have a repo containing the old single-file prototype, copy these files into it instead, and in step 1 also run `git rm RE-SOURCE-AI-prototype*.html` (or keep it under `/prototype` for history).

---

## Step 1 — Initialize project with README, package.json and .gitignore

Project skeleton: README describing every feature, `package.json` with `npm start` / `npm test`, and `.gitignore`.

```bash
git add .gitignore package.json README.md
git commit -m "chore: initialize project with README, package.json and .gitignore"
```

## Step 2 — Add page markup and dark industrial theme styles

All page markup (login gate + 7 pages) and the full stylesheet. Opens in a browser but nothing is interactive yet.

```bash
git add index.html css/styles.css
git commit -m "feat(ui): add page markup and dark industrial theme styles"
```

## Step 3 — Add trained RandomForest model and CODD dataset images

The trained forests (`data/model.json`, ~950 KB, 100 trees) and real CODD images pulled out of the prototype's base64 blobs into proper `.jpg` files.

```bash
git add data/model.json assets/
git commit -m "feat(data): add trained RandomForest model and CODD dataset images"
```

## Step 4 — Add material constants, second-life uses and demo market listings

Lookup tables: material labels/colours, weight + CO₂ + ₹ factors, pathway → use mapping, condition-tiered second-life uses, training baselines, demo buyer listings.

```bash
git add js/config/
git commit -m "feat(config): add material constants, second-life uses and demo market listings"
```

## Step 5 — Add RandomForest inference engine with tests

Pure decision-tree inference — material type, condition, recovery potential, pathway. First tests: **5 passing**.

```bash
git add js/ml/forest.js js/ml/model-loader.js tests/forest.test.js
git commit -m "feat(ml): add RandomForest inference engine with tests"
```

## Step 6 — Add in-browser pixel feature extraction with tests

Sobel edges, Otsu segmentation, HSV colour, Laplacian texture, circularity, solidity → the 16-feature vector. Tests: **10 passing**.

```bash
git add js/ml/features.js tests/features.test.js
git commit -m "feat(ml): add in-browser pixel feature extraction with tests"
```

## Step 7 — Add perceptual-hash reference photo matching

dHash + Hamming distance against hand-labelled reference photos, with an `ENABLE_REFERENCE_MATCH` switch.

```bash
git add js/ml/reference-match.js
git commit -m "feat(ml): add perceptual-hash reference photo matching"
```

## Step 8 — Add shared state, session metrics and utilities with tests

Shared state with a passport-change event (so features don't import each other), one metrics function for dashboard/ESG/export, hashing, CSV download, geo. Tests: **16 passing**.

```bash
git add js/state.js js/metrics.js js/utils/ tests/metrics.test.js
git commit -m "feat(core): add shared state, session metrics and utilities with tests"
```

## Step 9 — Add hash router and animated home stats

Hash-based navigation between the 7 pages and the animated $227.4B / 40% / 12.2% counters.

```bash
git add js/features/router.js
git commit -m "feat(nav): add hash router and animated home stats"
```

## Step 10 — Add site-access login gate

"Badge in" screen — name + role, attributed on every passport entry and the ESG report.

```bash
git add js/features/login.js
git commit -m "feat(auth): add site-access login gate"
```

## Step 11 — Add digital material passport with JSON/CSV export

Checksummed `RSA-2026-0001-XXXX` IDs, verification pattern, detail modal, remove, JSON + CSV export.

```bash
git add js/features/passport.js
git commit -m "feat(passport): add digital material passport with JSON/CSV export"
```

## Step 12 — Add photo upload, box selection and AI material analysis

The core feature: upload / drag-drop / sample photo → draw a box → scan HUD → AI material detection → condition, recovery, pathway, second-life uses, correction + recalculate.

```bash
git add js/features/audit.js
git commit -m "feat(audit): add photo upload, box selection and AI material analysis"
```

## Step 13 — Add live recovery dashboard with circularity and value metrics

Recovery score ring, weight and CO₂ diverted, pathway mix bars, recovery vs. true circularity vs. EU 12.2%, ₹ value, donut chart, live activity feeds.

```bash
git add js/features/dashboard.js
git commit -m "feat(dashboard): add live recovery dashboard with circularity and value metrics"
```

## Step 14 — Add buyer matching with GPS/IP location sorting

Ranks buyers by strong / possible match, then real distance after GPS (or IP fallback) location; contact modal pre-fills the email with your matching passport IDs.

```bash
git add js/features/marketplace.js
git commit -m "feat(marketplace): add buyer matching with GPS/IP location sorting"
```

## Step 15 — Add BRSR-aligned ESG impact report with print to PDF

Session summary, circularity vs benchmark, material table, BRSR Principle 6 note; Download/Print button saves as PDF.

```bash
git add js/features/esg.js
git commit -m "feat(esg): add BRSR-aligned ESG impact report with print to PDF"
```

## Step 16 — Add how-it-works page, training gallery and sample photos

Training-set gallery, ground-truth image, and the 6 sample site-photo buttons on the Audit page.

```bash
git add js/features/about.js
git commit -m "feat(about): add how-it-works page, training gallery and sample photos"
```

## Step 17 — Wire all features together in app bootstrap

`main.js` loads the model and initialises every feature. **The app is fully working from this commit.**

```bash
git add js/main.js
git commit -m "feat(app): wire all features together in app bootstrap"
```

## Step 18 — Run unit tests on every push with GitHub Actions

GitHub Actions runs `npm test` on every push and pull request.

```bash
git add .github/
git commit -m "ci: run unit tests on every push with GitHub Actions"
```

---

## Push

```bash
git push -u origin main
```

Check `git status` shows *nothing to commit, working tree clean* before pushing. Then on GitHub open the **Actions** tab — the `tests` workflow should go green.

## Make it live (GitHub Pages — free)

1. Repo → **Settings → Pages**
2. Source: **Deploy from a branch** → Branch: `main`, folder: `/ (root)` → **Save**
3. After ~1 minute it's live at `https://<your-username>.github.io/resource-ai/`

Put that link in the hackathon submission and the repo's *About* box.

## Tips

- Pushing all 18 at once is fine — GitHub keeps each commit separate. If you want the history spread across the team, each teammate can make some of the commits on their own machine (pull before each one).
- To see the history: `git log --oneline`
- Made a typo in the last commit message? `git commit --amend -m "new message"` (only before pushing).
