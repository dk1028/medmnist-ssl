# Week 1 Feedback

* **Clear experiment config**: dataset, model, finetune mode, epochs, batch size, LR, and weight decay are all reported.
* **Included calibration**: providing ECE and a reliability diagram is a strong Week 1 habit (most people stop at accuracy).
* **Baseline mindset**: treating this as a comparison point for later improvement is exactly what is for

---

## Results recap

* **Accuracy**: 0.7051
* **AUROC**: 0.5441
* **ECE**: 0.1084

---

## Main concern: Accuracy (0.705) vs AUROC (0.544) mismatch ⚠️

Seeing decent accuracy but near-random AUROC usually indicates one of these two issues:

### 1) Class imbalance “accuracy illusion”

BreastMNIST is typically imbalanced (often roughly 1:3 malignant:benign). A model that predicts the majority class most of the time can still get ~0.70 accuracy.

**Action items**

* Report a majority-class baseline accuracy (what you get if you always predict the most common class).
* Add confusion matrix, balanced accuracy, F1, and/or precision/recall.
* Consider reporting per-class recall (especially for the minority class).

### 2) Possible AUROC computation bug (very common)

AUROC must be computed from continuous scores/probabilities, *not* hard class predictions.

**Quick AUROC checklist**

* Use probability scores:

  * If using 2-logit output: `p = softmax(logits, dim=1)[:, 1]`
  * If using 1-logit output (sigmoid): `p = sigmoid(logit)`
* Ensure AUROC is computed on `p` (shape `[N]`) with labels `y` (shape `[N]`).
* Confirm **which label is “positive”** (malignant=1 vs benign=1). A flipped definition can make interpretation confusing.
* Confirm `model.eval()` + `torch.no_grad()` during evaluation.
* Confirm labels are correct shape/type (`y.squeeze(1).long()` if MedMNIST gives `[B,1]`).

> **Recommendation:** Before tuning hyperparameters, first verify AUROC is computed correctly using probability scores.

---

## About “underfitting”

It’s reasonable to suspect underfitting, but **test metrics alone can’t confirm it**.

**To diagnose under/overfitting**, include:

* Train vs validation loss curves
* Train vs validation accuracy/AUROC across epochs

Suggested phrasing for the README:

> “These baseline results are low; next we will inspect train/val curves to determine whether the model is underfitting or overfitting, and tune accordingly.”

---

## Calibration note (ECE = 0.108)

* **ECE ~0.11** is not catastrophic, but calibration interpretation is limited if AUROC is near-random.
* The reliability diagram suggests predictions are mostly in **mid-to-high confidence** bins; after fixing AUROC/probability handling, it will be more meaningful to compare calibration across models.

---

## High-impact next steps (priority order) 

1. **Add imbalance-aware reporting**

   * Majority baseline accuracy
   * Confusion matrix
   * Balanced accuracy, F1, per-class recall

2. **Verify AUROC pipeline**

   * Ensure AUROC uses probability scores (`softmax[:,1]` or `sigmoid`), not argmax labels.
   * Confirm positive class definition.

3. **Train longer / log learning curves**

   * `epochs=5` is usually short for a ResNet baseline.
   * Try 20–50 epochs with validation monitoring.

4. **Compare finetune modes**

   * Compare `FINETUNE='head'` vs `FINETUNE='all'`.
   * If finetuning all layers, consider **smaller LR for backbone** and larger LR for head (differential LR).

5. **Handle class imbalance (optional but often helpful)**

   * Weighted cross-entropy, focal loss, or weighted sampler.

---

## README improvements (Week 1 expectations)

To make the Week 1 report more complete, add:

* **Preprocessing details**: input size, channel conversion (grayscale→3ch), normalization.
* **Data split + class counts** in train/val/test.
* **Learning curves** (even just a small table of epoch metrics).
* Keep the GitHub setup issue brief (OK to mention), but prioritize **model/data-related** issues.

---

## Summary

You built a solid Week 1 baseline and included calibration, which is great. The key next step is to resolve the **accuracy vs AUROC mismatch** by (1) reporting imbalance-aware metrics and (2) confirming AUROC is computed from **probability scores** with the correct positive class. After that, longer training and finetuning comparisons should yield clear improvements in Week 2/3.
