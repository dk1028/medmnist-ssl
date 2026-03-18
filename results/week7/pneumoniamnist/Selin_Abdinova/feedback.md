Thanks for the Week 7 submission, the ROC/operating-point part looks directionally good, but the error-analysis section needs to be fixed before the conclusions can be trusted.

# What’s strong

The ROC curve looks consistent with a reasonably strong classifier: it rises steeply at low FPR and appears compatible with the Week 6 AUROC range.

Your two operating points are sensible and interpretable:

TPR ≈ 0.90 at threshold 0.894 with FPR = 0.162

FPR ≈ 0.10 at threshold 0.970 with TPR = 0.826

The written takeaway is also reasonable: this is exactly the kind of threshold trade-off we want to discuss in Week 7.

# Main issue: the error gallery looks incorrect / duplicated

The biggest problem is the attached P4_error_gallery.png.

The same false-positive image appears repeated multiple times across the top row.

The same false-negative image also appears repeated multiple times across the bottom row.

All FP scores are shown as 1.00, and all FN scores as 0.06, which makes the duplication issue even more suspicious.

This strongly suggests that the gallery code is repeatedly selecting the same index/image rather than the top-k distinct errors.

# Please check the gallery generation code for:

sorting errors correctly but then accidentally indexing the same first element multiple times,

not removing already-selected indices,

using a scalar instead of an array when looping,

or reusing the same image tensor inside the plotting loop.

# P4 notes are a bit too generic right now

Because the gallery appears duplicated, the current notes:

“False positives often show…”

“False negatives tend to…”

are not yet well-supported by the figure.

Once the gallery is fixed, please make the comments more specific:

say how you selected the examples
(“top-5 distinct false positives by predicted pneumonia probability”)

mention whether the images were taken from the test set

and describe actual visible patterns in those specific examples, not just generic possibilities.

# P2 needs one more reproducibility detail

Please clarify:

were these thresholds chosen on the validation set and then evaluated on test,

or chosen directly on the test ROC?

For Week 7, threshold selection should ideally be done on validation, then reported on test.
Right now the markdown gives the threshold values, but not the selection protocol.

# Small improvements that would make this much stronger

Add AUC value directly on the ROC figure (or in the markdown next to the operating points).

Add one small table with:

threshold,

TPR,

FPR,

precision,

specificity,

accuracy
for each operating point.

If you want to connect to Week 6, you can explicitly say:

“The high-confidence errors are consistent with the calibration issue observed previously.”

#  Suggested resubmission checklist

Fix the error-gallery code so it shows distinct FP/FN examples

State the threshold selection protocol (val → test)

Add AUC and optionally one small operating-point summary table

Rewrite the P4 notes after the gallery is corrected
