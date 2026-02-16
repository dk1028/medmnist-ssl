# What’s strong

Clear, correct setup for a linear probe: frozen SSL encoder (ResNet-18) + trained linear head only.

Good documentation of key preprocessing choices: 224×224 resizing, grayscale → 3-channel, and normalization.

Using stratified sampling for 5% and 10% label subsets is a solid choice for a medical dataset and helps reduce label imbalance artifacts.

Results are reported cleanly in both README and JSON, and the trend is sensible: 10% > 5% on both accuracy and AUROC.

# Results interpretation (what the numbers suggest)

Your probe shows meaningful label efficiency: AUROC improves from 0.756 → 0.862 when moving from 5% to 10% labels.

Compared to the Week 3 fully supervised baseline (Acc 0.83, AUROC 0.961), the probe is still behind, which is expected, but it would be useful to clarify whether that baseline used 100% of labels and whether it had the same backbone + resolution.

# Key improvements to make this more rigorous / fair

Make the baseline comparison apples-to-apples.
Right now you compare:

Probe trained on 5%/10% labels
vs

Week 3 supervised baseline likely trained on 100% labels (and possibly different augmentation/training details).

Actionable fix: Add at least one supervised baseline at the same label fractions (5% and 10%), for example:

Supervised training from scratch (or ImageNet init) using 5%/10% labels
This is the most direct way to show whether SSL helps label efficiency.

Report variability (seeds) or confidence intervals.
With small labeled sets (n=236, n=472), results can vary noticeably with the random seed.

Actionable fix: Run 3 seeds and report mean ± std for Acc/AUROC.

Add one more metric or class breakdown (optional but high value).
For medical classification, AUROC is good, but it helps to include:

sensitivity/recall for the positive class, specificity, or a confusion matrix
This shows whether improvements come from better detection of pneumonia cases.

Clarify probe training details (for reproducibility).
The README is missing training hyperparameters.

# Actionable fix: Add 1or 2 lines:

optimizer, learning rate, epochs, batch size, weight decay, and loss function.

# Suggested stronger “Takeaways” phrasing

Your takeaway is fine but could be more informative. For example:

“Linear probing shows strong gains from 5% to 10% labels (AUROC 0.756 → 0.862), indicating the SSL encoder provides useful transferable features. However, the probe still underperforms a fully supervised model trained with full labels (AUROC 0.961). A fairer next comparison is supervised training at the same 5%/10% label budgets to quantify SSL’s label-efficiency advantage.”
