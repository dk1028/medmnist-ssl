# Week 7 

## Overview

Week 7 builds on the Week 6 fine-tuned SSL encoder (20% labels) to evaluate threshold selection, ranking performance via ROC and PR curves, and high-confidence failure modes via an error gallery. All analysis uses the best model from Week 6: SSL Finetune 20% (AUROC 0.813, ECE 0.112).


## ROC and Precision-Recall Curves

| Metric  | Value |
|---------|-------|
| ROC-AUC | 0.813 |
| PR-AUC  | 0.920 |

The ROC curve shows strong discrimination well above the random baseline, with a sharp early rise in TPR at low FPR,  indicating that the model confidently identifies most malignant cases before making many false positive errors.

The PR curve is particularly informative here given the class imbalance (positive class base rate of ~ 0.73). PR-AUC of 0.920 substantially exceeds the random baseline of 0.73, and precision remains high (>0.90) across most of the recall range. The steep drop at high recall reflects the unavoidable precision-recall tradeoff at the extremes. In a screening context, PR-AUC is arguably more meaningful than ROC-AUC since it directly captures performance on the positive (malignant) class.

---

## Operating Points

Thresholds were selected on the **validation set** and evaluated on the **test set**.

| Operating Point | Threshold | TPR  | FPR  | Accuracy | Balanced Acc | TP | FP | TN | FN |
|-----------------|-----------|------|------|----------|--------------|----|----|----|----|
| Target TPR=0.90 | 0.538     | 0.842| 0.429| 0.769    | 0.707        | 96 | 18 | 24 | 18 |
| Target FPR=0.10 | 0.983     | 0.228| 0.024| 0.430    | 0.602        | 26 |  1 | 41 | 88 |

**Target TPR=0.90** — achieved actual TPR of 0.842 on the test set, slightly below the 0.90 target due to val/test distribution shift. This point catches 96 of 114 malignant cases but accepts 18 false alarms (FPR=0.43). In a clinical screening context this is likely the more appropriate operating point, since missing a malignancy is more costly than a false alarm that triggers a follow-up.

**Target FPR=0.10** — achieved FPR of 0.024 (extremely conservative) but TPR collapses to 0.228, missing 88 of 114 malignant cases. This operating point is unsuitable for screening, since a model that misses 77% of positive cases provides little clinical value despite its precision.


## High-Confidence Error Gallery

Errors collected at the TPR=0.90 operating point (threshold=0.538).

### False Positives — Predicted Malignant, Actually Benign

| Index | P(malignant) | Notes |
|-------|-------------|-------|
| 85    | 0.986       | Features dark pixels from the bottom of the sample, and a slightly lighter hole in the center |
| 44    | 0.978       | Features a shadowed jagged hole in the center of the sample, similar to malignant cases  |
| 48    | 0.945       | Mottled texture of uneven brightness, dark pixels around the border may have flagged it malignant |
| 141   | 0.931       | Few lesions, dark hole reching towards the center of the sample, visually ambiguous|
| 127   | 0.923       | Fewer lesions, with some speckles on the right side of the sample, jagged holes and shadowbehind the sample consistent with malignant cases |

### False Negatives — Predicted Benign, Actually Malignant

| Index | P(malignant) | Notes |
|-------|-------------|-------|
| 5     | 0.039       | low contrast lesions, very few dark pixels characteristic of malignant cases |
| 51    | 0.071       | Features smooth boundaries and a more regular lesion pattern |
| 76    | 0.113       | Well defined margins, uncharacteristic of malignancy, though the sample does have a large sample characteristic of malignancy |
| 41    | 0.193       | Borderline case, appears malignant to me, the sample features jagged boundaries, though the dark areas are wider than the usual malignant case |
| 64    | 0.214       | Borderline case, visually ambiguous |

**Patterns:** False positives cluster at very high confidence (0.92–0.99), suggesting the model is systematically overconfident on certain benign cases with irregular or heterogeneous textures. False negatives tend to have very low predicted probabilities (0.04–0.21), indicating the model is not just uncertain but actively predicts benign — these are likely well-circumscribed malignant lesions that share visual features with benign cases.



## Conclusion

ROC-AUC of 0.813 and PR-AUC of 0.920 confirm the SSL fine-tuned encoder generalizes well beyond a fixed threshold. The TPR=0.90 operating point is the more clinically appropriate approach for screening, accepting a higher false positive rate to minimize missed malignancies. Error analysis reveals two distinct failure modes: high-confidence false positives on heterogeneous benign lesions, and low-confidence false negatives on well-defined malignant cases — both suggesting the model relies heavily on boundary regularity and texture contrast as proxies for malignancy.
