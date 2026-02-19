# Reference: SimCLR – Milder Augmentations for Controlled Comparison

**Reference:**  
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.  
[arXiv:2002.05709](https://arxiv.org/abs/2002.05709)

---

## Summary

**Key idea:**  
SimCLR demonstrates that the strength of colour distortion and blur directly affects representation quality, if its too weak the model relies on colour histograms as a shortcut. If it's too strong the augmentations may destroy structure. Takeaway: The optimal strength is dataset‑dependent.

**Protocol/metric:**  
- Ablation studies on augmentation parameters: the paper systematically varies jitter strength, grayscale probability, and blur to measure their impact on linear evaluation accuracy.  
- They report that even moderate distortions (e.g., jitter strength 0.2) still outperform no distortion, but the full recipe gives best results.

**How I used it:**  
**Model 2** deliberately reduces the strength of colour jitter (`0.2` instead of `0.4`), lowers grayscale probability (`p=0.1`), and narrows blur (`[0.1,0.25]`). This creates a controlled comparison: we assess whether medical images (breastmnist) benefit equally from the full SimCLR recipe or require gentler augmentations. Downstream metrics (accuracy, AUROC, ECE) are compared between Model 1 and Model 2.

**Note** This is still incomplete, I intend to do at least 2 more models, and do some comparisons with GaussianNoise

---
