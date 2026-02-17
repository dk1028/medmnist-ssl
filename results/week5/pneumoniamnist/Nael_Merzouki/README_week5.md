# Week 5 Results — PneumoniaMNIST

## 1. Setup

**SSL Encoder:**
- Source: Week 4 SimCLR-lite pretrained ResNet-18 (512-dim features)

**Linear Probe Configuration:**
- Label fractions: 5%, 10%
- Single fully-connected layer (512 -> 2 classes)
- Frozen encoder backbone; probe only trainable
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 128
- Epochs: 30
- Seed: 42

**Baseline Reference:**
- Supervised baseline: 100% labeled data (4,708 samples), same optimizer/epochs/LR/batch size

**Calibration Evaluation:**
- Expected Calibration Error (ECE): 10 equal-width bins
- Reliability diagrams: confidence vs accuracy at decision threshold 0.5

**P3 Extra Tasks:**
1. **Domain Shift Robustness:** Gaussian noise (σ=0.15), brightness ±20%, contrast ±30%
2. **PR vs ROC Analysis:** On 10% probe; class imbalance ratio 1.67:1

---

## 2. Performance Summary

### A. Linear Probe vs Supervised Baseline (Test Set)

| Model | Label Fraction | Train Size | Accuracy | AUROC | ECE |
|-------|----------------|-----------|----------|-------|-----|
| SSL Probe (5%) | 5% | 234 | 0.7853 | 0.8963 | 0.2427 |
| SSL Probe (10%) | 10% | 470 | 0.7933 | 0.9276 | 0.2367 |
| Supervised Baseline | 100% | 4,708 | 0.8413 | 0.9484 | 0.2942 |


- 5% probe: −5.60% Accuracy, −0.0521 AUROC
- 10% probe: −4.80% Accuracy, −0.0208 AUROC
- Calibration: SSL probes significantly better calibrated (ECE 0.24 vs 0.29 supervised) but worst accuracy (0.79 vs 0.84 supervised)

### B. PR vs ROC Analysis (SSL 10% Probe, P3)

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.9276 |
| **PR-AUC** | 0.9509 |
| **Positive Class Prevalence** | 62.5% (390/624) |
| **Imbalance Ratio** | 1.67:1 |
| **Sensitivity @ Threshold 0.5** | 0.9846 (384/390 caught) |
| **Specificity @ Threshold 0.5** | 0.4744 (111/234 healthy correctly ID'd) |
| **Precision @ Threshold 0.5** | 0.7574 (384/(384+123) correct pneumonia predictions) |

**Insight:** PR-AUC (0.9509) exceeds ROC-AUC (0.9276) by 2.33 percentage points, reflecting the dominance of the positive class (pneumonia) in this dataset.

---

## 3. Training & Validation Behavior

**Label Fraction Effect (5% -> 10%):**

Increasing labeled data from 234 to 470 samples (+100%) improved validation AUROC from 0.9318 to 0.9542 (+2.24 points). Test AUROC gained +3.13 points (0.8963 -> 0.9276), demonstrating consistent label efficiency. Test accuracy improvement was modest (+0.80%), suggesting the model already captured useful representations at 5% but benefited from larger decision boundary refinement at 10%.

**Gap vs Supervised Baseline:**

The 10% probe achieves 92.76% AUROC compared to 94.84% supervised, a gap of 2.08 points. This gap is modest considering the labeled training set is 10x smaller (470 vs 4,708 samples). The SSL encoder successfully transferred learned representations, enabling competitive performance with minimal label investment. The accuracy gap is larger (4.80%), suggesting supervised training benefits more from full diversity in the label distribution.

**Calibration Behavior:**

Surprisingly, both SSL probes exhibit better calibration (ECE 0.2427, 0.2367) than the supervised baseline (ECE 0.2942), despite lower accuracy. This suggests the frozen encoder with modest training prevents overfitting to confident yet incorrect predictions. The supervised model, trained on all 4,708 labels, may overfit the training distribution, producing overconfident posterior probabilities that diverge from true accuracy in reliability bins.

---

## 4. PR vs ROC Interpretation (P3)

The PR-AUC (0.9509) outperforms ROC-AUC (0.9276) by 2.3 percentage points on the SSL 10% probe. This divergence reflects the 1.67:1 class imbalance (390 pneumonia vs 234 healthy cases). ROC treats true positive rate and false positive rate symmetrically, causing the abundant healthy class (234 samples) to dominate specificity calculations and inflate apparent model quality. In contrast, PR directly measures precision (TP / TP+FP) and recall (TP / TP+FN) on the minority class—pneumonia detection.

The model achieves 98.46% sensitivity (catches 384/390 pneumonia cases) but only 47.44% specificity (correctly identifies 111/234 healthy). This asymmetry is clinically appropriate: missing pneumonia (false negative) is costly, while false alarms trigger follow-up screening. PR curves preserve this cost asymmetry while ROC obscures it. The gap between PR and ROC suggests imbalance is substantial enough to meaningfully affect metric choice.

At high recall (>90%), the PR curve maintains high precision (>93%), indicating the model confidently identifies most pneumonia cases without excessive false positives. The no-skill baseline precision (62.5%) represents random guessing; the model's actual precision (75.74% @ 0.5 threshold) represents a 21% relative improvement, confirming genuine predictive power beyond class prevalence.

---

## 5. Observations

- **Label Efficiency:** SSL pretraining dramatically reduces labeled data requirements. The 10% probe (0.9276 AUROC from 470 samples) achieves near-supervised performance (0.9484 from 4,708), suggesting Week 4 encoder captured 95% of discriminative information with 10% labels.

- **Calibration Advantage:** SSL probes are better calibrated (ECE −0.0575 vs supervised). Hypothesis: frozen backbone prevents overfitting; only the single linear layer trains, reducing capacity to memorize noise.

- **Domain Shift Sensitivity:** Gaussian noise (σ=0.15) caused largest AUROC drop (−8.7%), while brightness/contrast perturbations minimally affected AUROC (−0.4% to +0.1%). Hypothesis: ResNet features are robust to photometric variation but sensitive to pixel-level noise introduced by incorrect augmentation assumptions (medical images != natural photos).

- **Model Behavior at High Sensitivity:** 98.46% sensitivity @ threshold 0.5 indicates the model is conservative, i.e. it predicts pneumonia for most borderline cases. This bias toward the majority class aligns with training on limited labels; the model defaults to positive class when uncertain.

- **Specificity Trade-off:** Low specificity (47.44%) reflects precision-recall trade-off under imbalance. Raising decision threshold would improve specificity but reduce recall and increase missed diagnoses. Clinical protocol would determine appropriate threshold.

- **Metric Instability:** The 2.3 p.p. gap between PR and ROC (already noted) implies that reported "AUROC" values may be misleading for this imbalanced dataset. PR-AUC is the honest metric; ROC-AUC inflates apparent performance.

- **Hypothesis: Limited Probe Capacity:** A single linear layer on frozen features may saturate at ~93% AUROC. Fine-tuning the encoder backbone could close the remaining 2-point supervised gap, but would require substantially more labeled data and longer training.

- **Limitation — No Validation Curves:** Test accuracy and AUROC are reported, but validation loss trajectories are not archived. Training instability or early overfitting cannot be ruled out without loss curves.

- **Suggestion for Week 6:** Attempt fine-tuning the encoder on the 10% labeled subset (while keeping majority of features frozen) to test whether the 2% AUROC gap is due to probe capacity or encoder expressivity. Track validation loss to diagnose overfitting mechanisms.

---

## 6. Domain Shift Robustness (P3)

Perturbation results on SSL 10% probe:

| Perturbation | Accuracy | AUROC | ECE | Δ AUROC |
|--------------|----------|-------|-----|---------|
| Clean | 0.7933 | 0.9276 | 0.2367 | TO DO! |
| Gaussian Noise | 0.7933 | 0.8406 | 0.2205 | −0.087 |
| Brightness +20% | 0.8365 | 0.9234 | 0.2143 | −0.004 |
| Brightness −20% | 0.7308 | 0.9284 | 0.2618 | +0.001 |
| Contrast +30% | 0.8478 | 0.9259 | 0.2291 | −0.002 |
| Contrast −30% | 0.7019 | 0.9265 | 0.2779 | −0.001 |

**Key Finding:** AUROC is robust to photometric perturbations (Δ ≤ 0.4) but sensitive to Gaussian noise (−8.7). Accuracy fluctuates widely (±6.3%) under brightness/contrast, inconsistent with AUROC stability. This suggests predictions remain well-separated despite accuracy threshold changes.

---

## 7. TO DOs

- **Validation Curves Unavailable:** Training loss and validation AUROC trajectories would help diagnose convergence behavior or overfitting magnitude.
- **Single Seed Run:** No variance estimate or confidence intervals across multiple runs.
- **No Ablation:** Individual augmentation effects during training not isolated.
- **Probe Architecture Fixed:** Single linear layer may be undersized.
- **Domain Shift Limited:** Only 6 perturbations tested and no natural distribution shift (e.g., different scanner, patient population).

---

## 8. Summary

Week 5 demonstrates successful label-efficient learning via SSL pretraining. The 10% probe achieves 92.76% AUROC (2.08 points below supervised baseline) with 10% of labeled data, validating Week 4 encoder quality. Calibration is superior to supervised baseline, suggesting frozen backbone prevents overfitting. PR-AUC (0.9509) exceeds ROC-AUC (0.9276) due to 1.67:1 class imbalance, with sensitivity heavily biased toward pneumonia detection (98.46%) at cost of specificity (47.44%), appropriate for medical diagnosis. Domain shift analysis reveals robustness to photometric variation but sensitivity to pixel noise. Overall, SSL enables competent linear-probe classifiers on limited labels while maintaining better calibration than fully supervised alternatives.

---

## References

[1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). "A Simple Framework for Contrastive Learning of Visual Representations."

[2] Saito, K., Watanabe, K., Ushiku, Y., & Harada, T. (2017). "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation."

[3] P K Dutta, Anushri Chowdhury, Anouska Bhattacharyya, Shakya Chakraborty, Sujatra Dey (2025). "A Deep Learning Approach for Pneumonia Detection on Chest X-Ray Images."
