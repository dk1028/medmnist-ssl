Thanks for the Week 7 submission - the structure and intent are good (operating point + ROC/PR + error gallery).
I really like that you included (1) an explicit operating point choice, (2) ROC/PR plots, (3) a high-confidence error gallery, and (4) a short narrative about FP/FN patterns. That’s exactly the right direction for Week 7.

However, there are major consistency issues in the metrics/plots that must be fixed before we can trust the conclusions.

However, there are major consistency issues in the metrics/plots that must be fixed before we can trust the conclusions.

# Critical issues (must fix)
# 1) ROC/PR curves contradict the reported model performance

Your roc_curve.png shows AUC = 0.3986 and the curve is mostly below the diagonal, which implies the model is worse than random (or labels/scores are flipped).

But in Week 6 and in your own summary table you report AUROC ≈ 0.9524 for “SSL FineTune (10%).”

These cannot both be true for the same model + same test split.
➡️ Please regenerate ROC/PR curves from the exact same predictions used to compute AUROC=0.9524, and ensure the plotted AUC matches the numeric report.

Common causes to check:

using the wrong probability column (probs[:,0] vs probs[:,1])

using predicted class labels (0/1) instead of continuous scores/probabilities for ROC/PR

pos_label mismatch (like pneumonia class encoded as 0 but treated as 1)

accidentally evaluating a different checkpoint/model

# 2) error_analysis.md metrics disagree with README + plots

error_analysis.md reports: Accuracy 0.6234, FPR 1.0, ROC AUC 0.3986, etc.

README claims for the same model at threshold 0.5: Accuracy 0.8814, AUROC 0.9524, and a confusion matrix that doesn’t align with those accuracy values.
You need one single source of truth: generate all numbers (confusion matrix, accuracy, AUROC, PR-AUC, precision/recall) from the same evaluation script and save them together.

# 3) Confusion matrix vs reported accuracy mismatch

You list: TP=471, FP=255, TN=267, FN=31 (total 1024).
That implies accuracy = (TP+TN)/total = (471+267)/1024 ≈ 0.721, not 0.8814.
So either the confusion matrix is wrong, or the accuracy you copied is from a different run/split.


# Operating point selection (needs improvement)

Choosing threshold=0.50 is fine as a baseline, but Week 7 usually expects operating points chosen based on a goal (screening vs confirmatory use).

Please add a threshold sweep table (thresholds 0.1 → 0.9) with:

sensitivity/recall, specificity, precision, FPR, FNR, accuracy, (optional) balanced accuracy

Then pick at least two operating points:

Screening point: target high sensitivity (e.g., ≥ 0.95)

High-specificity point: target fewer false positives (e.g., specificity ≥ 0.90)
Importantly: choose thresholds on validation, then report final metrics on test.

Right now, operating_points_table.csv only contains threshold=0.5, so it doesn’t support your threshold discussion yet.

# Error gallery: good start, but make it more rigorous

The “high-confidence FP/FN” idea is great. To strengthen it:

State explicitly how you select them (like “top-5 FP by predicted probability among incorrect test predictions”).

Make sure images are saved in a human-readable scale (de-normalize before saving), because some examples look extremely low-contrast / artifact-like.

Add a quick summary count: how many FP/FN total at the chosen operating point, not just top examples.
