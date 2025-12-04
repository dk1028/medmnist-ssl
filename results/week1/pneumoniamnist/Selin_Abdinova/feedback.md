Good Week 1 baseline and nice debugging. The pipeline runs end-to-end, the label-shape bug is correctly fixed, and you already report accuracy, AUROC, and ECE. Performance is still modest (AUROC ≈ 0.65), but calibration looks reasonably good (ECE ≈ 0.11)

Issue & fix documented clearly: You correctly identified the RuntimeError: 0D or 1D target tensor expected as a label-shape issue and fixed it with
y = y.squeeze(1).long() in train/val/test. That’s exactly the right solution for MedMNIST labels.

# Improvements
Add a one-line interpretation of the reliability plot.
For example:
“The reliability diagram shows the model is roughly calibrated (ECE ≈ 0.11), with only mild over/under-confidence in some confidence bins.”
