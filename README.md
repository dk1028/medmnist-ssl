# medmnist-ssl — Weeks 1–10 README (Research Plan + How-To)

> **Goal:** Undergraduate-friendly, **research-oriented** project using **MedMNIST2D (binary-class only)** with a required **Self-Supervised Contrastive Learning (SSL)** phase. We will build baselines, pretrain with SSL, and evaluate label efficiency and calibration.

* Site: [https://medmnist.com/](https://medmnist.com/)
* Allowed Week 1 datasets: **BreastMNIST** (`breastmnist`), **PneumoniaMNIST** (`pneumoniamnist`)
* Repo name: **`medmnist-ssl`** 

---

## 0) Roles & Dataset Assignment

* Each member **must choose one** MedMNIST2D **binary** dataset: `breastmnist` or `pneumoniamnist`.
* You will be **assigned** to your choice and keep it for all weeks (for fair comparisons).
* Create your personal results folder under:
  `results/weekX/<dataset_key>/<your_name>/`

**How to claim a dataset**

* Create a GitHub Issue titled: `Signup: <your_name> — <dataset_key>`
* Include: system specs (Colab/local), preferred model (smallcnn/resnet18)

---

## 1) Repository Structure

```
medmnist-ssl/
  ├─ notebooks/
  │   └─ ML_Basics_MedMNIST_Colab.ipynb
  ├─ starter/
  │   ├─ requirements.txt
  │   └─ src/
  │       ├─ data.py
  │       ├─ metrics.py
  │       ├─ models.py
  │       ├─ train_medmnist.py
  │       └─ utils.py
  ├─ results/
  │   ├─ week1/ ... week10/
  │   │   └─ <your_name>/
  ├─ reports/
  ├─ figures/
  ├─ README.md
  └─ .gitignore
```

**.gitignore (minimum)**

```
runs/
__pycache__/
.ipynb_checkpoints/
*.pt
*.png
*.json
```

> Commit only curated artifacts to `results/` (metrics/plots/screens). Do **not** commit bulky raw `runs/`.

---

## 2) Environment & Tools

* **Primary runtime:** Google Colab (GPU). Local runs are optional.
* **Packages:** `torch`, `torchvision`, `medmnist`, `scikit-learn`, `matplotlib`, `tqdm`, `numpy` (already in the notebook/starter).
* **Data handling:** The `medmnist` package **auto-downloads** datasets on first use and caches at `~/.medmnist` (Colab: `/root/.medmnist/`).

**Colab quickstart**

1. Open `notebooks/ML_Basics_MedMNIST_Colab.ipynb` in Colab.
2. **Runtime → Change runtime type → GPU**.
3. Run cells top→down. In the config cell, set:

```python
DATASET_KEY = 'pneumoniamnist'  # or 'breastmnist'
MODEL_NAME  = 'resnet18'        # or 'smallcnn' if slow
EPOCHS      = 5                 # adjust per week
BATCH_SIZE  = 128               # use 64/32 if memory is tight
```

**Local quickstart (optional)**

```bash
pip install -r starter/requirements.txt
python -m src.train_medmnist --dataset breastmnist --model smallcnn --epochs 5
# or
python -m src.train_medmnist --dataset pneumoniamnist --model resnet18 --epochs 5 --finetune head
```

---


> This project uses **MedMNIST2D (binary)** only for Week 1 and beyond. Each member must choose **one** binary dataset — **BreastMNIST** (`breastmnist`) or **PneumoniaMNIST** (`pneumoniamnist`) — and will be assigned to it for the remainder of the project.

* Allowed keys: `breastmnist`, `pneumoniamnist`

---

## A. Quickstart on Google Colab (Recommended)

1. Open the notebook and enable GPU: **Runtime → Change runtime type → GPU**.
2. Install dependencies:

   ```python
   !pip -q install medmnist torch torchvision scikit-learn matplotlib tqdm
   ```
3. Auto‑download and inspect a binary dataset (first run creates a cache):

   ```python
   import medmnist
   from medmnist import INFO

   KEY = 'pneumoniamnist'   # or 'breastmnist'
   DataClass = getattr(medmnist, INFO[KEY]['python_class'])

   train_ds = DataClass(split='train', download=True)
   val_ds   = DataClass(split='val',   download=True)
   test_ds  = DataClass(split='test',  download=True)

   print('Cache root:', train_ds.root)          # typically /root/.medmnist on Colab
   print('Shape:', train_ds.imgs.shape)
   print('Labels:', INFO[KEY]['label'])
   ```

**Notes**

* Colab cache location: `/root/.medmnist`. Subsequent calls won’t re‑download.
* If memory is tight, reduce image size (e.g., 32×32) or batch size.

---

## B. Local Setup (Optional)

1. Create a virtualenv (recommended) and install packages:

   ```bash
   pip install medmnist torch torchvision scikit-learn matplotlib tqdm
   ```
2. Auto‑download the dataset to your home cache:

   ```python
   import medmnist
   from medmnist import INFO

   KEY = 'breastmnist'      # or 'pneumoniamnist'
   DataClass = getattr(medmnist, INFO[KEY]['python_class'])

   train = DataClass(split='train', download=True)
   val   = DataClass(split='val',   download=True)
   test  = DataClass(split='test',  download=True)

   print(train.root)  # usually ~/.medmnist
   ```
3. (Optional) Add Torch transforms and DataLoaders exactly as in the Colab example.

**Notes**

* Local cache location: `~/.medmnist` (Windows/macOS/Linux under your home directory).
* To force a re‑download, delete the cache directory and re‑run with `download=True`.

---

## C. Dataset Keys & Labels (Binary 2D Only)

```python
from medmnist import INFO
for key in ['breastmnist', 'pneumoniamnist']:
    print(key, INFO[key]['label'])  # class names
```

* `breastmnist`: benign vs malignant
* `pneumoniamnist`: pneumonia vs normal

> We stick to binary tasks to stabilize the pipeline early (clean baselines, fair comparisons), then extend with SSL in Weeks 4–6.

---

## D. Where to Place Artifacts in This Repo

After a run (Colab or local), copy curated results (not entire `runs/`) to:

```
results/week1 to 10/<dataset_key>/<your_name>/
  ├─ test_metrics.json
  ├─ reliability.png
  ├─ screenshot_env.png
  └─ README_week1.md
```


## 3) Week-by-Week Plan (Deliverables Included)

### Week 1 — Kickoff & Baseline Run (binary only)

**Do:**

* Pick `breastmnist` or `pneumoniamnist` (via Issue signup).
* Colab GPU on → run 5-epoch baseline (`smallcnn` or `resnet18 --finetune head`).
* Verify outputs: `best.pt`, `val_log.jsonl`, `test_metrics.json`, `reliability.png`.

**Submit to** `results/week1/<dataset>/<your_name>/`:

* `test_metrics.json`, `reliability.png`, `screenshot_env.png`, `README_week1.md` (3–5 sentences: model/epochs/batch; one issue + fix; Acc/AUROC note).

---

### Week 2 — EDA & Baselines

**Do:**

* Quick EDA: class counts, sample tiles, input sizes.
* Train `smallcnn` and `resnet18 --finetune head` (5–8 epochs). Record Val Acc & Macro-AUROC.

**Submit (week2/…):**

* Table: model vs Acc/AUROC; 3–5 misclassified samples with comments.

---

### Week 3 — Finetune-All & Light Augmentations

**Do:**

* Compare `resnet18 --finetune head` vs `--finetune all`.
* Light augmentation ablation (2–3 variants) appropriate to modality.

**Submit (week3/…):**

* Learning curves (train/val), ablation table (variant → AUROC), short note on over/underfitting.

---

### Week 4 — SSL Pretraining (Required)

**Do:**

* Implement **SimCLR-lite** or **MoCo-lite** on the train split treated as unlabeled.
* Two augmented views per image; contrastive loss; small projection head.

**Suggested settings:**

* SimCLR: large batch if possible; MoCo-lite: queue 2k–8k, momentum ~0.99.
* Temperature τ ≈ 0.2; AdamW lr 1e-3 to 3e-4; 30–50 epochs on Colab.

**Submit (week4/…):**

* SSL training log (loss vs epoch), config JSON (τ, batch/queue, augs), short paragraph explaining choices.

---

### Week 5 — Linear Probe & Label Efficiency

**Do:**

* Freeze SSL encoder; train a linear head with **1% / 5% / 10%** labels (keep Val/Test unchanged).

**Submit (week5/…):**

* Plots: fraction vs Accuracy; fraction vs Macro-AUROC.
* Comparison note vs supervised-only and ImageNet-transfer baselines.

---

### Week 6 — SSL Fine-tuning & Calibration

**Do:**

* Unfreeze encoder; small-lr fine-tune; compute **ECE** & plot **reliability diagram**.

**Submit (week6/…):**

* Table: SSL probe vs SSL fine-tune vs baselines (Acc, AUROC, **ECE**).
* Reliability plot + 3–5 line interpretation.

---

### Week 7 — Thresholds, PR/ROC, Error Analysis

**Do:**

* Choose operating points (e.g., target TPR/FPR).
* Class-wise PR (binary case: positive class PR), collect **high-confidence errors**.

**Submit (week7/…):**

* Threshold table; error gallery with brief annotations and potential fixes.

---

### Week 8 — Ethics, Limits, and Mini Robustness Test

**Do:**

* Short note on risks/assumptions; optional MC Dropout uncertainty on a batch, or tiny domain-shift test.

**Submit (week8/…):**

* 1-page note (ethics/limits/mitigations); small table/plot for the mini test.

---

### Week 9 — Writing & Figures

**Do:**

* Draft a short, paper-style writeup (4–6 pages): Intro, Methods, Experiments, Results, Discussion.
* Finalize label-efficiency, ROC/PR, reliability plots; tidy tables.

**Submit (week9/…):**

* Draft v1 PDF + checklist of gaps; figure sources and scripts.

---

### Week 10 — Talk & Final Artifacts

**Do:**

* Prepare a 10–12 min talk; two practice runs.
* Freeze artifacts; top-level README pointers to reproduce in ≤1 hour on Colab.

**Submit (week10/…):**

* Final PDF (writeup) + slide deck; reproducibility checklist.

---

## 4) Metrics & Evaluation

* **Primary:** Accuracy, **Macro-AUROC** (binary: AUROC), **ECE** (reliability diagram).
* **Secondary:** Precision/Recall/F1 (report for transparency).
* Keep seeds fixed; note all hyperparameters and label fractions.

---

## 5) Contribution Workflow

* Use Issues for questions (`WeekX: <name> <dataset>`).
* Commit messages: `Week3: finetune-all + aug ablation (breastmnist)`.
* Prefer PRs for larger changes; include brief descriptions and screenshots.

---

## 6) Troubleshooting (Colab)

* **No GPU** → Runtime → Change runtime type → GPU; rerun install cell.
* **Slow** → `MODEL_NAME='smallcnn'`, lower `BATCH_SIZE`.
* **Import errors** → rerun install cell (Colab resets on restart).
* **Save errors** → ensure the output directory exists.

---

## 7) Ethics, Privacy, Integrity

* No PHI; use only public MedMNIST data under its license.
* Report results honestly; include failure cases; avoid overclaiming.

---

## 8) Success Criteria

* Clean baselines and SSL results at 1%, 5%, 10% labels on your assigned **binary** dataset.
* Clear evidence of when SSL helps (or not) with calibration analysis.
* Reproducible artifacts, tidy repo, concise writeup and talk.
