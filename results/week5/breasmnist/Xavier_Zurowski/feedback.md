

Overall, this is a really good. The linear-probe setup is correct (frozen SSL encoder + linear classification head), your reporting is clean (README tables match the JSON metrics), and you used appropriate metrics for class imbalance (Balanced Accuracy + AUROC). The plot also makes the key comparisons easy to see.

# Key items to clarify / fix (important for correctness & fairness):

Baseline definition is ambiguous (“Supervised (head only, frozen enc)”).
Please explicitly state what encoder is being frozen for this baseline:

frozen randomly-initialized encoder?

frozen ImageNet-pretrained encoder?

frozen SSL-pretrained encoder?
Right now it’s unclear, and that affects how meaningful the comparison is. A single sentence in the README would resolve this.

AUROC for the head-only baseline looks suspicious given the collapse signal.
Balanced Accuracy = 0.500 strongly suggests majority-class prediction / degenerate behavior, but AUROC = 0.439 is unusual in that scenario. Please double-check your AUROC computation, especially:

Are you passing probabilities/scores (sigmoid(logits) or softmax probability for the positive class), not hard labels?

Are you using the positive-class probability (proba[:, 1] for 2-class softmax)?

Is pos_label oriented correctly (not flipped)?

# Reporting / reproducibility improvements (quick wins):

Add minimal probe training details: optimizer, learning rate, epochs, batch size, weight decay.

Describe label fraction sampling: random seed, whether sampling is stratified by class.

Mention whether thresholding was fixed at 0.5 or tuned on validation.

Suggested next step to strengthen the supervised baseline:
Since BreastMNIST is imbalanced and your head-only baseline appears to collapse, try one simple fix so the baseline doesn’t degenerate:

weighted loss (e.g., pos_weight for BCEWithLogitsLoss), or

class-balanced sampler, or

threshold tuning on validation for balanced accuracy.

This will make the comparison more informative and increase confidence that your SSL representations are truly driving the gain.

# Minor edits:

Typo fixes: “represnatation” → “representation”, “BreasMNIST” → “BreastMNIST”

Remove duplicate “## Analysis” header.

The plot y-axis label could be “Score” instead of “Accuracy” since it includes AUROC.
