# P3 — PR vs ROC under Class Imbalance (PneumoniaMNIST)

In PneumoniaMNIST, the dataset is moderately imbalanced, with the positive class (pneumonia) more frequent than the negative class. Under imbalance, ROC-AUC can sometimes give an overly optimistic view of performance because it measures ranking quality across all thresholds without directly reflecting precision.

PR-AUC focuses on precision (positive predictive value) and recall, which makes it more sensitive to false positives when evaluating a medical screening model. In clinical settings, false positives can carry meaningful cost (unnecessary follow-up imaging or treatment), so PR curves are often more informative.

In this experiment:

- ROC-AUC = 0.487  
- PR-AUC = 0.626  

These results are poor overall because the fine-tuning stage was limited to a single training epoch due to compute constraints. As a result, the model did not sufficiently separate classes, and predicted probabilities were clustered near 0.5, leading to weak ranking performance (ROC-AUC ≈ 0.5).

Under proper training (multiple epochs), we would expect:
- ROC-AUC to increase as ranking improves.
- PR-AUC to provide a clearer view of precision–recall trade-offs at clinically meaningful thresholds.

Therefore, the current curves primarily illustrate methodological differences between ROC and PR rather than strong model performance.
