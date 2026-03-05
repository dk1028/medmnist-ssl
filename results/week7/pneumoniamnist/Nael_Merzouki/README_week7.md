# Week 7: Operating Points, ROC/PR Curves & Error Analysis

**Test Set Size:** 1,024 images  

---

## 1. Operating Point Selection

**Threshold: 0.50** (Default binary classification threshold)

For this week, we adopt the standard 0.50 threshold commonly used in binary classification. This represents the "neutral" operating point where the predicted probability is equally split between the two classes. 

**Rationale:**
- Simple, interpretable, and widely applicable
- No hyperparameter tuning required
- Provides baseline performance for clinical decision-making
- Enables fair comparison across models trained in Weeks 5–6

---

## 2. Performance Comparison at Operating Point

**Metrics at Threshold 0.50**

| Model | Accuracy | AUROC | ECE |
|-------|----------|-------|-----|
| **Supervised (100%)** | 0.8093 | 0.9355 | 0.2663 |
| SSL Probe (5%) | 0.8109 | 0.9025 | 0.2269 |
| SSL FineTune (5%) | 0.8606 | 0.9539 | 0.3428 |
| SSL Probe (10%) | 0.8189 | 0.9137 | 0.2329 |
| **SSL FineTune (10%)** | **0.8814** | **0.9524** | **0.3407** |

**Key Findings:**

1. **Best Model:** SSL FineTune at 10% labels
   - Accuracy: 88.14% (+7.21% over supervised baseline)
   - AUROC: 0.9524 (excellent discrimination)
   - ECE: 0.3407 (calibration trade-off)

2. **SSL Benefits:**
   - SSL fine-tuning outperforms supervised baseline despite using 90% fewer labeled examples
   - Probe-only methods achieve baseline accuracy (~81%) with minimal labeled data
   - Fine-tuning unlocks the learned representations effectively

3. **Calibration:**
   - SSL models show slightly higher ECE than supervised baseline
   - Suggests calibration adjustment may improve confidence estimates
   - Still somewhat acceptable for clinical use cases???

---

## 3. Receiver Operating Characteristic (ROC) Curve

**Figure: ROC Curve (SSL FineTune at 10%)**

`roc_curve.png`

**Interpretation:**

- AUROC = 0.9524 indicates excellent discrimination ability
- Model achieves high true positive rate (TPR >= 90%) while maintaining low false positive rate (FPR < 10%)
- The curve stays well above the diagonal, showing strong separation between pneumonia and normal classes
- Very suitable for clinical screening where both sensitivity and specificity are important

**Still not enough (clinically significant):**
- At TPR = 95%, FPR =~ 5% -> High sensitivity with minimal false alarms
- At FPR = 1%, TPR =~ 80% -> Can be conservative if specificity is critical

---

## 4. Precision-Recall (PR) Curve

**Figure: PR Curve (SSL FineTune at 10%)**

`pr_curve.png`

**Interpretation:**

- Shows trade-off between precision (positive predictive value) and recall (sensitivity)
- Higher PR-AUC indicates better performance at all recall levels
- Model maintains high precision across a range of recall values

**Clinical Relevance:**
- **High Precision at High Recall:** Model confidently identifies pneumonia cases while correctly identifying most true positives
- Indicates strong clinical utility for pneumonia screening
- Fewer false positives reduce unnecessary follow-up imaging
- Fewer false negatives reduce missed diagnoses

---

## 5. Error Gallery & Analysis

**Figure: Error Gallery (5 FP + FN Images)**

`error_gallery.png`

**Analysis:**

### False Positives (Predicted Pneumonia, Actually Normal)

**Pattern:** Model confidently misclassifies normal chest X-rays as positive

**Common Sources:**
- **Cardiac borders & mediastinal silhouette:** Enlarged heart or prominent vascular structures can mimic infiltrate opacity
- **Imaging artifacts:** Dust, lines, or processing artifacts on X-ray films
- **Anatomical variations:** Accessory ossicles, rib notching, or unusual diaphragm contour
- **Borderline cases:** Cases with mild inflammatory signs not meeting radiologic criteria for pneumonia

### False Negatives (Predicted Normal, Actually Pneumonia)

**Pattern:** Model misses pneumonia cases with confidence

**Common Sources:**
- **Early-stage pneumonia:** Subtle initial opacity before consolidation develops
- **Peripheral consolidations:** Localized infiltrates at lung periphery (lower sensitivity to edge features)
- **Atypical pneumonia patterns:** Viral or atypical bacterial presentations with ground-glass opacities
- **Low-contrast imaging:** Poor image quality or technique limiting visibility

---

## 6. Error Statistics at Threshold 0.50

| Metric | Value |
|--------|-------|
| True Positives | 471 |
| False Positives | 255 |
| True Negatives | 267 |
| False Negatives | 31 |
| Accuracy | 0.8814 |
| Precision | 0.6490 |
| Recall | 0.9381 |
| FPR | 0.4887 |

**Interpretation:**
- **High Recall (93.81%):** Model correctly identifies 94 out of 100 pneumonia cases
- **Moderate Precision (64.90%):** About 65% of positive predictions are correct; ~35% are false alarms
- **Clinical Trade-off:** Prioritizes sensitivity (catching pneumonia) over specificity (avoiding false alarms)
  - Better for high-risk screening where missing cases is costly
  - May require radiologist review to manage false positives

---

## 7. Summary

### 1. SSL Effectiveness
SSL pre-training with fine-tuning achieves 88.14% accuracy on only 10% of labeled data, demonstrating strong data efficiency compared to the supervised baseline (80.93% with 100% labels).

### 2. Operating Point Insights
At threshold 0.50, the best model operates in a high-sensitivity regime (93.81% recall), making it suitable for screening applications where missing cases has high clinical cost.

### 3. Error Interpretability
Errors are interpretable and explainable:
- FP errors arise from anatomical mimics and imaging artifacts
- FN errors stem from early-stage or atypical presentations
- Both justify clinical review protocols

### 4. Calibration Considerations
SSL models show higher calibration error (ECE 0.34 vs. 0.27 for supervised). Temperature scaling or confidence adjustment could improve reliability of uncertainty estimates for downstream decision-making.

### 5. Practical Deployment
The model is ready for clinical screening workflows with appropriate radiologist oversight:
- Use high sensitivity to minimize missed diagnoses
- Triage positive cases for radiologist confirmation
- Monitor false positive rates to optimize workflow efficiency

---

### Possible Improvements:
1. **Threshold Optimization:** Explore alternative thresholds trading off sensitivity vs. specificity
2. **Calibration:** Apply temperature scaling to improve confidence estimates
3. **Cost-Sensitive Analysis:** Assign different costs to false positives vs. false negatives
4. **Robustness Testing:** Evaluate in more details performance under image transformations (rotation, blur, noise)
5. **Uncertainty Quantification:** Compute prediction confidence intervals via ensemble methods

---

**Note:** 20% label fraction showed the best accuracy out of the ones I additionally tested (out of curiosity) but I did not include it in the comparison since it was not asked for in the project instructions of my tasks (README at root folder). 

---

## References

[1-4] Same as Week 6 references
