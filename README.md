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
results/week1 to 10/<your_name>/
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


**Week 1 summary:**
Breast: Low discrimination (especially AUROC) vs. good calibration (low ECE)

Pneumonia: Good discrimination (high AUROC) vs. overconfidence (high ECE)
---

## Week 2 — EDA, Baselines & First Calibration Check

> **Theme:** "Know your data, trust your baselines."  This week you (1) explore the dataset, (2) build **two solid supervised baselines**, and (3) run a **first calibration / reliability check** that we will refine in later weeks.

Target datasets (fixed from Week 1):

* `breastmnist`  (benign vs malignant)
* `pneumoniamnist` (pneumonia vs normal)

All work stays on **your assigned dataset**.

---

### 2.1 Goals

By the end of Week 2, you should:

1. Understand the **basic statistics** of your dataset:

   * Class counts (train/val/test)
   * Example images per class
   * Input shape / value range
2. Train **two supervised baselines** on your dataset:

   * `smallcnn`
   * `resnet18 --finetune head`
3. Produce a **short, interpretable comparison**:

   * Accuracy and **Macro-AUROC** for both models
   * A few **misclassified examples** with comments
   * A *first* look at **calibration** (reliability diagram + ECE)

This sets the reference point for Weeks 3–6 (finetuning, SSL, and calibration improvements).

---

### 2.2 Directory & Filenames

Create your personal folder for this week:

```bash
results/week2/<dataset_key>/<your_name>/
```

Inside, we expect something like:

```text
results/week2/<dataset_key>/<your_name>/
  ├─ metrics_smallcnn.json
  ├─ metrics_resnet18_head.json
  ├─ cm_smallcnn.png             # confusion matrix
  ├─ cm_resnet18_head.png
  ├─ misclassified_gallery.png   # 3–5 interesting mistakes
  ├─ reliability_smallcnn.png    # optional but strongly encouraged
  ├─ reliability_resnet18_head.png
  └─ README_week2.md             # your 0.5–1 page summary
```

> As always, do **not** commit raw `runs/` directories. Only commit curated metrics/plots/screenshots.

---

### 2.3 Step 1 — Quick EDA

Use either the provided Colab notebook or a small Python script in `starter/src/` to:

1. **Inspect shapes and dtypes**

   * Print `imgs.shape`, `imgs.dtype`, and label dictionary `INFO[KEY]['label']`.
2. **Compute class counts**

   * For train/val/test, count how many samples belong to each label.
   * Report the **class balance** in `README_week2.md`.
3. **Visualize sample tiles**

   * Show at least 4–8 examples per class.
   * Comment on obvious differences (e.g., contrast, intensity, texture).

In your README, add 2–4 bullet points answering:

* Is the dataset **balanced or imbalanced**?
* Are there any **surprising artifacts** (e.g., text overlays, padding)?
* Do images for different classes look clearly separable to you?

---

### 2.4 Step 2 — Supervised Baselines (smallcnn vs resnet18-head)

We standardize two baselines:

1. **Small CNN** (from `models.py`):

   * For fast iterations and sanity checks.
2. **ResNet-18, linear head only (`--finetune head`)**:

   * Backbone frozen, train only the classifier head on MedMNIST.

Suggested hyperparameters (tweak if needed, but keep notes):

* Epochs: **5–8**
* Batch size: 64–128 (reduce if you hit OOM)
* Optimizer: Adam or SGD with momentum (as in starter script)

Example CLI calls:

```bash
# smallcnn
python -m src.train_medmnist \
  --dataset <dataset_key> \
  --model smallcnn \
  --epochs 8

# resnet18, finetune head only
python -m src.train_medmnist \
  --dataset <dataset_key> \
  --model resnet18 \
  --finetune head \
  --epochs 8
```

For each run, save test/validation metrics into separate JSON files
(fore example, `metrics_smallcnn.json`, `metrics_resnet18_head.json`).
If the script already outputs a metrics JSON, simply copy/rename it into your Week 2 folder.

In `README_week2.md`, include a **small table** like:

| Model           | Val Acc | Test Acc | Test AUROC |
| --------------- | ------: | -------: | ---------: |
| smallcnn        |    0.xx |     0.xx |       0.xx |
| resnet18 (head) |    0.xx |     0.xx |       0.xx |

and a 2–3 sentence comment on which model behaves better and why you think that is.

---

### 2.5 Step 3 — First Calibration / Reliability Check

This week we already start to **look at calibration**, but only lightly. A full calibration study comes later (Weeks 5–6).

1. For at least one model (preferably `resnet18 --finetune head`):

   * Collect **predicted probabilities** on the **validation** or **test** set.
   * Build a **reliability diagram** and compute **ECE**.

2. **Binning strategy**

   * Default: 10 **equal-width** bins on [0, 1].
   * *Improvement tip:* Try **equal-frequency bins** (each bin has roughly the same number of samples). This often makes the reliability plot easier to interpret in high-confidence regions.

3. **Plotting**

   * Plot the **mean predicted probability** vs **empirical accuracy** per bin.
   * Also show the **histogram of sample counts** per bin (either in the same figure or a small inset) so readers can see where most predictions lie.
   * In the title or caption, **print the final ECE value** (3 decimal places), e.g., `ECE = 0.073`.

If `metrics.py` already has an ECE implementation, use it. Otherwise, you can implement a simple version for now; it does not need to be perfect at this stage.

In `README_week2.md`, write 3–5 sentences interpreting the plot, for example:

* Does the model tend to be **over-confident** (predicted probabilities too high)?
* Or **under-confident** (probabilities too low)?
* Is calibration different across low vs high confidence bins?

> **Small dataset-specific infos (optional, not required):**
>
> * For **PneumoniaMNIST**, a single round of **temperature scaling (TS)** on validation logits often reduces ECE noticeably, while AUROC barely changes. You can try TS once if you have time and compare ECE before/after.
> * For **BreastMNIST**, spending more effort on **better features** (for example, slightly longer training, light data augmentation) may matter more than TS at this stage.

If you do try TS, save a second reliability plot (for example, `reliability_resnet18_head_TS.png`) and briefly note the difference.

---

### 2.6 Misclassified Examples & Error Commentary

Pick **3–5 interesting misclassified samples** from the test set for one of your models (preferably the stronger one).

For each sample, record:

* True label vs predicted label
* Predicted probability for the predicted class
* A one-sentence hypothesis: *Why* might the model have failed here?

Examples of comments:

* "Borderline intensity; looks visually closer to the other class."
* "Very low contrast; global brightness much darker than typical images."
* "Possible artifact / text overlay interfering with the pattern."

Arrange these into a small figure (`misclassified_gallery.png`) or a simple table in `README_week2.md`.

This kind of **qualitative error analysis** will be revisited in Week 7 when we study thresholds and high-confidence errors more systematically.

---

### 2.7 What to Write in `README_week2.md`

Aim for **0.5–1 page** with clear structure. A suggested template:

1. **Dataset recap (2–4 bullets)**

   * Class balance
   * Any noticeable artifacts or quirks
2. **Baseline comparison (short paragraph + table)**

   * Mention which model is stronger and any over/underfitting signs
3. **Calibration snapshot (3–5 sentences)**

   * ECE value(s), one or two key observations from the reliability diagram
   * If you tried TS, one sentence on how ECE changed and what stayed the same (e.g., AUROC)
4. **Error analysis (bulleted list)**

   * 3–5 bullet points summarizing what the misclassified examples suggest about model limitations or dataset challenges

Keep the tone **lab notebook + mini-report**: clear, honest, and focused on what you learned, not just the numbers.

---

### 2.8 Checkpoint: Are You Ready for Week 3?

You are ready to move on when:

* [ ] You can **describe your dataset** (class balance and typical appearances).
* [ ] You have **two working supervised baselines** with recorded metrics.
* [ ] You’ve generated at least one **reliability diagram** and an ECE value.
* [ ] You have looked at **actual misclassified images** and written down at least a few observations.

These pieces will be directly reused when we compare `--finetune all`, augmentations (Week 3), and SSL representations (Weeks 4–6).

---

### Week 3 — Finetune-All & Light Augmentations

> **Theme:** "Same backbone, different training regimes." This week you will (1) compare **head-only** vs **finetune-all** ResNet-18, and (2) run a **small augmentation ablation** that is realistic for your modality.

All work stays on **your assigned dataset** (`breastmnist` or `pneumoniamnist`).

---

## 3.1 Goals

By the end of Week 3, you should be able to:

1. Run **two ResNet-18 regimes** on your dataset:

   * **Head-only** (frozen backbone, train classifier head) — your Week 2 baseline.
   * **Finetune-all** (all layers trainable).
2. Add **light, medically reasonable data augmentations** on the **train** split only.
3. Read **learning curves** (train vs val) and diagnose **overfitting / underfitting**.
4. Decide on a **recommended supervised baseline config** that we will carry into SSL (Weeks 4–6).

> Think of Week 3 as: "Which supervised setup should we trust before we invest time in SSL pretraining?"

---

## 3.2 What We Assume from Weeks 1–2

Before starting Week 3, you should already have (on your assigned dataset):

* A working **data pipeline** (MedMNIST loader, DataLoader, transforms).
* At least **two supervised baselines** from Week 2:

  * `smallcnn`
  * `resnet18` with **head-only** training (backbone frozen)
* Basic metrics on the **test** set:

  * Accuracy, AUROC (and maybe ECE)
* At least one **reliability diagram** and some notes about calibration.

If any of these are missing or broken, fix them first; you will reuse this code for multiple weeks.

---

## 3.3 Week 3 Folder & Expected Artifacts

Create your personal Week 3 folder:

```text
results/week3/<dataset_key>/<your_name>/
```

Inside, we expect something like:

```text
results/week3/<dataset_key>/<your_name>/
  ├─ metrics_resnet18_head.json          # head-only (you can copy from Week 2)
  ├─ metrics_resnet18_all_basic.json     # finetune-all, basic transforms
  ├─ metrics_resnet18_all_augA.json      # finetune-all, augmentation config A
  ├─ metrics_resnet18_all_augB.json      # (optional) augmentation config B
  ├─ curves_resnet18_head.png            # train/val curves
  ├─ curves_resnet18_all_basic.png
  ├─ curves_resnet18_all_augA.png
  ├─ aug_ablation_table.md (optional)    # or put the table directly in README
  └─ README_week3.md                     # 0.5–1.5 page summary
```

> As always, **do not commit** raw `runs/` or very large logs. Only commit **curated** metrics, plots, and a short README.

---

## 3.4 Step 1 — Baseline: ResNet-18 Head-Only (Frozen Backbone)

We keep the Week 2 **head-only** ResNet-18 as our reference point.

1. Either **reuse** your Week 2 head-only run or **re-run** it with the same hyperparameters:

   ```bash
   # Example: head-only finetuning (backbone frozen)
   python -m src.train_medmnist \
     --dataset <dataset_key> \
     --model resnet18 \
     --finetune head \
     --epochs 8
   ```

2. Make sure you have a **metrics JSON** (test set) with at least:

   * `acc`, `auroc` (and `ece` if available)

3. If your training script logs **per-epoch metrics** (train/val loss or accuracy), use them to plot a simple **learning curve**:

   * x-axis: epoch
   * y-axis: accuracy or loss
   * one line for **train**, one for **val**

4. Save the plot as:

   ```text
   curves_resnet18_head.png
   ```

If your current script does not save per-epoch metrics yet, you can:

* Use the notebook version and collect metrics in Python lists, or
* Add a few lines in `train_medmnist.py` to append `train_loss`, `val_loss`, `val_acc` to a list and save them as a small `.json` or `.csv`.

> We do **not** need a perfectly engineered logging system – a small dictionary or CSV is already useful.

---

## 3.5 Step 2 — Finetune-All ResNet-18 (All Layers Trainable)

Now we allow **all ResNet-18 layers** to update, not just the final classifier head.

Conceptually:

* **Head-only**: backbone parameters are **frozen**; only the last linear layer is trained.
* **Finetune-all**: backbone parameters are **unfrozen**; the entire network learns from MedMNIST.

In code, this usually means something like:

```python
# Pseudo-code sketch (do not copy blindly)
for param in backbone.parameters():
    param.requires_grad = True  # unfreeze everything
```

Or in the CLI, you may have a flag like `--finetune all` or simply **omit** `--finetune head` so that all layers train. Use whatever convention your script uses, but:

> **Label the run as `finetune-all` whenever the entire ResNet-18 is trainable.**

Run a finetune-all experiment with the **same basic transforms** you used so far (no new augmentations yet):

```bash
python -m src.train_medmnist \
  --dataset <dataset_key> \
  --model resnet18 \
  --finetune all \
  --epochs 8
```

(Adjust the actual flag names to match your script.)

Save:

* Test metrics → `metrics_resnet18_all_basic.json`
* Learning curve plot → `curves_resnet18_all_basic.png`

Try to keep **all other hyperparameters** (epochs, batch size, optimizer, LR) the same as in the head-only run, so the comparison is fair.

---

## 3.6 Step 3 — Light, Modality-Aware Data Augmentations

Now we add **small, realistic augmentations** on the **train** split only. The **val/test** splits should **never** use strong augmentations; they must stay as close as possible to the original data.

### 3.6.1 Suggested Augmentations

Below are suggestions. You can adapt them, but keep them **medically plausible**:

* For **PneumoniaMNIST (chest X-ray)**:

  * `RandomHorizontalFlip(p=0.5)`
  * `RandomRotation(degrees=10)` or `RandomAffine(degrees=10, translate=(0.05, 0.05))`
  * (Optional) small intensity jitter: `ColorJitter(brightness=0.1, contrast=0.1)`
  * **Avoid** vertical flips (they invert anatomy) and very large rotations.

* For **BreastMNIST (breast ultrasound)**:

  * `RandomHorizontalFlip(p=0.5)`
  * `RandomRotation(degrees=10)`
  * (Optional) slight Gaussian noise or small contrast jitter

Example (PyTorch) for a slightly stronger augmentation config:

```python
import torchvision.transforms as T

train_transform_augA = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    # Normalize(...) as before
])
```

You can define **two** configs:

* **Basic** (what you already used in Week 2): `Resize → ToTensor → Normalize`
* **AugA**: `Basic + light geometric aug` (flip + small rotation)
* (Optional) **AugB**: `AugA + very light intensity jitter`

### 3.6.2 Finetune-All + Augmentations

For each augmentation config, run **finetune-all ResNet-18** again:

```bash
# Example: finetune-all with AugA
python -m src.train_medmnist \
  --dataset <dataset_key> \
  --model resnet18 \
  --finetune all \
  --epochs 8 \
  --train-aug augA   # or a similar flag / config name in your script
```

Save for each variant:

* Test metrics → `metrics_resnet18_all_augA.json`, `metrics_resnet18_all_augB.json`
* Learning curves → `curves_resnet18_all_augA.png`, `curves_resnet18_all_augB.png`

> If you only have time for **one** augmentation config, do just **AugA** and label it clearly.

---

## 3.7 Step 4 — Learning Curves & Over/Underfitting

Now compare the learning curves across your variants:

1. For each of the following, inspect `train` vs `val` metrics over epochs:

   * `resnet18_head` (head-only)
   * `resnet18_all_basic` (all layers, basic transforms)
   * `resnet18_all_augA` (all layers, with augmentations)

2. Questions to answer (in words, not only numbers):

   * Does **finetune-all** reach higher **train accuracy** than head-only? What about **val accuracy / AUROC**?
   * Do you see **overfitting**? (for example, train acc → 1.0 but val acc stops improving or gets worse.)
   * Does **AugA** help reduce overfitting or improve test AUROC slightly?
   * Is training with augmentations **less stable** (noisy curves) or about the same?

You can place the curves side-by-side in your report, but even a short description is valuable.

---

## 3.8 Step 5 — Augmentation Ablation Table

Summarize your Week 3 runs in a small table. Example:

```markdown
| Model / Variant            | Finetune | Train Aug   | Test Acc | Test AUROC | Notes               |
|--------------------------- |----------|------------ |---------:|----------: |---------------------|
| ResNet-18 (head-only)      | head     | basic       |   0.76   |    0.84    | Week 2 baseline     |
| ResNet-18 (all, basic)     | all      | basic       |   0.78   |    0.85    | mild overfitting    |
| ResNet-18 (all, AugA)      | all      | flip+rot    |   0.79   |    0.86    | best test AUROC     |
| ResNet-18 (all, AugB)      | all      | flip+rot+jit|   0.78   |    0.85    | similar, more noisy |
```

You do **not** need to match these numbers; they are just an example of format and level of commentary.

If you have ECE or calibration plots for any of these variants, you can add an extra column or a sentence in the notes (optional for Week 3).

---

## 3.9 What to Write in `README_week3.md`

Aim for **0.5–1.5 pages** (short but structured). Suggested outline:

1. **Setup Recap (3–5 bullets)**

   * Your dataset and class balance.
   * Which transforms you used before Week 3 (basic pipeline).

2. **Head-Only vs Finetune-All (paragraph + mini-table)**

   * 3–5 sentences comparing metrics and curves.
   * Include a small table with at least: `Variant`, `Test Acc`, `Test AUROC`.

3. **Augmentation Ablation (paragraph + table)**

   * Describe each augmentation config (1–2 lines each).
   * Include the ablation table from Section 3.8.
   * Comment on which config seems best and why.

4. **Learning Curve Interpretation (3–6 sentences)**

   * Mention any clear signs of overfitting/underfitting.
   * Note if augmentations stabilize or destabilize training.

5. **Takeaways for SSL (2–3 bullets)**

   * Which supervised setup you will treat as your **reference** going forward.
   * Any surprises (e.g., small improvements, large gains, or no effect).

Keep the tone like a **lab notebook + mini-report**: honest, descriptive, and focused on what you learned.

---

## 3.10 Dataset-Specific Focus: PneumoniaMNIST vs BreastMNIST

In Week 3, **both teams follow the same overall pipeline** (head-only vs finetune-all, light augmentations, learning curves). Here we spell out the parts where the **focus differs by dataset**. Only include these in your `README_week3.md` if they are relevant to **your** dataset.

### 3.10.1 PneumoniaMNIST Team (Chest X-ray)

For the PneumoniaMNIST group, emphasize:

* **Augmentations to prioritize**

  * Start from: `RandomHorizontalFlip(p=0.5)` + small `RandomRotation(degrees=10)`.
  * You may also try **very light translation** (for example, `translate=(0.05, 0.05)`) to mimic slight changes in patient positioning.
  * Be careful with intensity jitter: use **small** `ColorJitter` only, and report if it hurts or helps.
  * Explicitly **avoid vertical flips** or huge rotations that would break the anatomical orientation.

* **Dataset-specific questions to answer** (in README):

  * Does finetune-all + augmentations mainly improve **AUROC** or **calibration** (ECE), or both?
  * Do augmentations reduce obvious overfitting (train acc → 1.0 while val acc stagnates)?
  * When you visually inspect a few high-confidence **false positives / false negatives**, do they look like borderline cases or clear mistakes?

* **Optional Pneumonia-specific note**

  * If you have time, briefly comment on whether the model seems to be **over-calling pneumonia** (many false positives) or **missing pneumonia** (false negatives), since this trade-off is clinically important.

Keep these comments short (3–6 sentences) and place them in a small subsection of your `README_week3.md` titled, for example, "PneumoniaMNIST-specific observations".

### 3.10.2 BreastMNIST Team (Breast Ultrasound)

For the BreastMNIST group, the focus is slightly different:

* **Augmentations to prioritize**

  * Geometric: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=10)` are usually safe.
  * You may try **very light Gaussian noise** or a tiny contrast jitter to mimic ultrasound variability.
  * Avoid extreme changes (large noise, heavy blurring, strong contrast shifts) that might erase subtle lesions.

* **Dataset-specific questions to answer** (in README):

  * Does **finetune-all** help more for BreastMNIST than it did for PneumoniaMNIST (according to class slides / shared results)? This dataset is small and noisy, so backbone adaptation might matter more.
  * Do augmentations **stabilize** training (less noisy curves) or mainly act as regularization (slightly lower train acc but better val metrics)?
  * Are the model’s errors mainly on **very dark/low-contrast** cases, or on images where lesions are small and hard to see?

* **Optional Breast-specific note**

  * If you plot a few misclassified examples, add 2–3 bullets about what might make them hard (for example, speckle noise, unclear boundaries) and whether different augmentations might help.

You can collect these under a small subsection such as "BreastMNIST-specific observations" in your `README_week3.md`.

---

## 3.11 Week 3 Checklist

You are ready for Week 4 (SSL pretraining) when:

* [ ] You have **head-only** and **finetune-all** ResNet-18 runs with metrics on your dataset.
* [ ] You have tried at least **one** realistic augmentation config on the train split.
* [ ] You have at least **one learning curve plot** per main variant.
* [ ] You wrote `README_week3.md` summarizing:

  * Supervised variants tested
  * How augmentations affected performance
  * Which configuration you recommend as your **baseline** going into SSL.

These decisions will help interpret later results when we compare **SSL encoders** vs **purely supervised** models and analyze label efficiency in Weeks 4–6.

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
