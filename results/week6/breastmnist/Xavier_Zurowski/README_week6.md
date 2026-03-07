# Week 6 – SSL Finetuning and Calibration

## Overview

For Week 6 I explored finetuning my pre-trained self-supervised encoder (from Week 4) on the BreastMNIST dataset with 10% and 20% labeled splits. The goal was to assess how finetuning affects model performance and calibration compared to simply linear probing.
Models were evaluated using accuracy, balanced accuracy, AUROC, and ECE. Reliability diagrams with per-bin sample counts were produced for each model to help visualize calibration quality.
## Method

**Linear Probe:** The SSL encoder was frozen and a linear head trained on the labeled subset using class-weighted cross-entropy. Threshold was tuned on the validation set by maximizing balanced accuracy. (Same process as week 5)
**Finetuning:** The full encoder was unfrozen and trained jointly with the head using separate learning rates (encoder 1e-4, head 1e-3) and weight decay 1e-4. Post-hoc temperature scaling was applied on validation logits, followed by threshold tuning on the scaled probabilities.

## Results

### Linear Probe

| Label Fraction | Accuracy | Balanced Accuracy | AUROC | ECE   |
|----------------|----------|-------------------|-------|-------|
| 5%             | 0.647    | 0.654             | 0.704 | —     |
| 10%            | 0.609    | 0.672             | 0.736 | 0.093 |
| 20%            | 0.712    | 0.712             | 0.788 | 0.144 |

The linear probe results suggest the SSL encoder captures meaningful class-relevant structure even without labels. AUROC rises from 0.704 to 0.788 as the labeled fraction increases, and balanced accuracy improves from 0.654 to 0.712. Notably, the 10% probe is the best-calibrated model overall (ECE 0.093), while the 20% probe degrades to ECE 0.144 — suggesting the linear head becomes overconfident as it sees more data even with the encoder frozen.

### Fine-Tuning

| Label Fraction | Accuracy | Balanced Accuracy | AUROC | ECE   |
|----------------|----------|-------------------|-------|-------|
| 10%            | 0.756    | 0.675             | 0.737 | 0.161 |
| 20%            | 0.724    | 0.706             | 0.813 | 0.112 |

### Supervised Baseline

| Model                      | Accuracy | Balanced Accuracy | AUROC |
|----------------------------|----------|-------------------|-------|
| Fully supervised (enc+head)| 0.897    | 0.854             | 0.928 |

---

## Analysis

**Fine-tuning only pays off at 20%.** At 10%, fine-tuning is flat vs the probe on AUROC (0.737 vs 0.736) and hurts calibration (ECE 0.161 vs 0.093) — with only ~55 labeled samples, the encoder overfits rather than adapts. At 20%, fine-tuning meaningfully improves AUROC (0.813 vs 0.788) and achieves the second best ECE of any model (0.112), beating the 20% probe baseline (0.144).

**Calibration is not monotonic.** The 10% probe is better calibrated than the 20% probe (0.093 vs 0.144), and fine-tuning at 20% recovers calibration (0.101) while fine-tuning at 10% makes it worse (0.161). This pattern suggests the linear head becomes overconfident with more labeled data, while end-to-end fine-tuning on sufficient data produces more evenly spread feature distributions.

**Temperature scaling had minimal effect** — learned temperatures were close to 1.0 across all runs, indicating the models were already near-calibrated after training with class-weighted loss.

### Reliability Diagrams

All bins contain n=15–16 samples (equal-frequency binning working as intended), so bin-level estimates are as reliable as the test set size allows.

**20% fine-tune (ECE 0.131):** The dominant pattern is systematic underconfidence, bars consistently lie above the diagonal across low-to-mid confidence bins, meaning the model's predicted probabilities are lower than the true positive rates. High-confidence bins (0.9–1.0) cluster tightly and track the diagonal well. The overall shape is smooth and monotonically increasing..

**10% fine-tune (ECE 0.161):** Predictions are more compressed into a narrower probability range, fewer bins are populated across the confidence axis, with most mass near 0.8–1.0. Low-to-mid bins again show underconfidence (bars above diagonal), but the pattern is less smooth, with more oscillation at mid-to-high confidence. This erratic behavior across bins is consistent with the higher ECE and reflects the encoder having less reliable feature distributions when fine-tuned on only ~55 samples.

---

## Conclusion

The best-calibrated model is the **10% linear probe (ECE 0.093)**. The best overall model is **20% fine-tune**, which achieves the highest AUROC (0.813) and competitive calibration (ECE 0.112). Fine-tuning is beneficial when enough labeled data is available at 10%, the linear probe is the better choice. A substantial gap remains between all SSL models and the fully supervised baseline (AUROC 0.928), suggesting the SSL representations are useful but task-specific adaptation via full supervision remains superior.





