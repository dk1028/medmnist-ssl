# Baseline comparison (smallCNN vs ResNet18)
Numbers

smallcnn
Acc = 0.72
AUROC = 0.61

resnet18 (fine-tuned)
Acc = 0.84
AUROC = 0.88

This is a very nice, interpretable comparison: ResNet18 gives a big gain in discrimination (AUROC 0.61 → 0.88) and accuracy. This is exactly what we expect: the smallCNN is a “sanity check” model, while ResNet18 benefits from stronger features and higher capacity.

In your README_week2.md (or future report), it would be good to explicitly say something like:

“ResNet18 substantially outperforms the smallCNN on BreastMNIST (ΔAUROC ≈ +0.27), suggesting that pre-trained features are very helpful for this dataset.”


# Confusion matrices and error types
Your confusion matrices are very informative, like both false positives and false negatives are reduced.

Sensitivity improves slightly, and specificity improves a lot.

This supports a nice qualitative statement:

“Compared to smallCNN, ResNet18 reduces both missed cancers (FN) and false alarms (FP), so it is better in both sensitivity and specificity.”

(Optional) If you haven’t done it yet, you could also compute TPR/TNR/precision(search what those are on the web) for each model and put them in a small table.


# Calibration and ECE
You computed reliability diagrams with two binning strategies (equal-width and equal-frequency), like that’s exactly what the Week 2 theory note was aiming for 👌

smallCNN:
ECE (equal-width) ≈ 0.203
ECE (equal-freq) ≈ 0.219

Confidences are mostly in a narrow mid-range band (around 0.5–0.6), and the model is not very discriminative anyway (AUROC ≈ 0.61).
Because of this, the calibration picture is quite noisy and not very meaningful — this is fine to mention explicitly.

ResNet18 (fine-tune all):
ECE (equal-width) ≈ 0.056
ECE (equal-freq) ≈ 0.081

Most predictions are in the high-confidence region (0.6–1.0).
The reliability curve is close to the diagonal but shows a bit of over-confidence in some bins, especially when using equal-frequency bins.

Nice things you did:
Showed the histogram / fraction of samples per bin — this makes the ECE much easier to interpret.
Reported the ECE values in the title and printed them below the figure.

What you could add in the README:
One or two sentences comparing the binning strategies, for example:
“For ResNet18, equal-frequency binning reveals that most predictions lie in high-confidence bins; ECE is slightly higher but still small (≈0.08), suggesting mild over-confidence but overall decent calibration.”


# Misclassified high-confidence examples
Your misclassified gallery (true/pred/conf in the title) is exactly what we wanted for Week 2:
All 5 examples are high-confidence errors (0.90–0.99), which is perfect for discussing calibration.
Visually, many of these patches look quite ambiguous (low contrast, fuzzy boundaries, structures that could be mistaken for lesions).

Suggestions for the text:
Add 3–5 short bullets in README_week2.md like:
“Several high-confidence false positives show bright regions that resemble malignant masses but might be benign tissue.”
“Some false negatives appear very low contrast; the lesion is hard to see even by eye.”
This will connect very nicely later to Week 7 (error analysis and thresholds).


# Small suggests
None of these are “wrong”; they are optional improvements:
Make explicit which ResNet variant you used.
The plots say ResNet18-all, so it looks like you fine-tuned the whole network.

For completeness, it would be nice to mention in the README:
“I fine-tuned all ResNet18 layers (not just the head).”
Link back to Week 1 theory.
You can add one short sentence like:
“These reliability diagrams implement the ECE definition from Week 1, with both equal-width and equal-frequency binning.”

Short overall conclusion for Week 2.
One small paragraph at the end:
“On BreastMNIST, a smallCNN baseline achieves moderate performance but is poorly calibrated and heavily biased towards the positive class. A fine-tuned ResNet18 improves both AUROC (+0.27) and calibration (ECE from ≈0.20 to ≈0.06), while reducing both false positives and false negatives. High-confidence errors tend to be visually ambiguous patches, suggesting that some cases remain genuinely difficult even for strong models.”
