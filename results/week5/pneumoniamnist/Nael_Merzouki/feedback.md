Week 5 PneumoniaMNIST results look strong, this is a very complete submission.
You provided a clear README, multiple metric JSONs/CSVs, and supporting plots (comparison, PR vs ROC curves, domain-shift robustness). The experimental setup is also consistent with a proper “SSL label-efficiency” evaluation (frozen encoder + linear probe at 5%/10% vs a fully supervised baseline).

# What’s strong

Excellent organization & reproducibility: Separate JSONs for probe (5%/10%), supervised baseline, PR/ROC metrics, and domain shift results makes it easy to audit.

Label-efficiency story is convincing: The 10% probe reaches AUROC 0.9276 vs supervised 0.9484 with 10× fewer labels (470 vs 4708). That’s exactly the kind of result we want from SSL pretraining.

Good calibration reporting: Including ECE + reliability diagrams is a big plus (most people skip this).

Nice extra analyses (P3): Domain shift perturbations + PR-vs-ROC discussion shows initiative beyond the basic requirement.

# Important fixes / clarifications (these matter for correctness)

PR vs ROC interpretation has a key class-imbalance mix-up

Your numbers show positive prevalence = 62.5% (390/624), meaning pneumonia (positive class) is the majority, not the minority.

In the README you wrote that PR focuses on the minority class and that the “abundant healthy class dominates specificity calculations” ,that’s not accurate here because healthy is 234, smaller than pneumonia 390.

# Suggested correction:

PR-AUC is sensitive to prevalence; the no-skill baseline PR precision is 0.625, so PR-AUC can look high partly because positives are common.

ROC-AUC is relatively less sensitive to prevalence, but still useful as a threshold-free ranking metric.

You can still argue PR is clinically meaningful (precision/recall tradeoffs), but please fix the “minority/majority” wording.

# A few claims are too strong / need softer wording

“PR-AUC is the honest metric; ROC-AUC inflates performance” → This is usually emphasized when positives are rare. Here positives are common, so rephrase more carefully:

“PR and ROC highlight different aspects; PR is helpful for understanding precision/recall tradeoffs and threshold behavior.”

Domain shift table has an unfinished value

In the domain shift table you wrote Clean ΔAUROC = TO DO — fill it as 0.0 (or remove the Δ column for clean).

# Methodology notes (good to mention for fairness)

Supervised baseline hyperparams may not be optimal

You matched optimizer/LR/epochs across models for fairness, which is fine, but note that the fully supervised baseline might benefit from different LR/schedule/augmentations than a linear probe.

If you want a stronger comparison, consider a light tuning for supervised (e.g., LR sweep or cosine decay) and report the best.

Calibration evaluation: threshold vs calibration

Using threshold 0.5 for confusion matrix is fine, but calibration should be described as confidence vs accuracy across bins (threshold isn’t the calibration definition).

Also note ECE depends on binning; your choice (10 equal-width bins) is reasonable, but mention it briefly in captions/README (you already did, nice).
