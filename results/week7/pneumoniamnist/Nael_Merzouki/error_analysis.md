# Week 7: Error Analysis

## High-Confidence Errors (Top 5 FP & 1 FN)

### False Positives (Predicted Pneumonia, Actually Normal)
Likely sources:
- Cardiac borders or vascular shadows mimicking infiltrates
- Imaging artifacts (dust, lines)
- Borderline cases with mild inflammatory signs

### False Negatives (Predicted Normal, Actually Pneumonia)
Likely sources:
- Early-stage pneumonia (subtle opacity)
- Peripheral or localized consolidations
- Atypical pneumonia patterns
- Low-contrast imaging conditions

## Metrics at Threshold 0.50

- Accuracy: 0.6234
- Precision: 0.6244
- Recall: 0.9974
- FPR: 1.0000
- ROC AUC: 0.3986
- PR AUC: 0.5757
