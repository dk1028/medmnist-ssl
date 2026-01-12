# Week 3 
## Setup Recap

**Dataset:** PneumoniaMNIST  
Binary classification of pediatric chest X-ray images:
- Label 0: Normal
- Label 1: Pneumonia


### Training Regimes
- **Head-only:** Backbone frozen, only final classification head trained
- **Finetune-all (basic):** All layers trainable, no augmentation
- **Finetune-all + AugA:** All layers trainable with light data augmentation

### Augmentation (AugA)
Applied **only to the training split**:
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=10)`

These augmentations are medically reasonable for chest X-rays and preserve anatomical plausibility.

---

## Head-only vs Finetune-All Comparison

| Model Variant | Accuracy | AUROC | ECE |
|---------------|----------|-------|-----|
| ResNet18 (head-only) | 0.7628 | 0.8252 | 0.2389 |
| ResNet18 (finetune-all, basic) | 0.6218 | 0.6494 | 0.1127 |
| ResNet18 (finetune-all, AugA) | **0.8558** | **0.9703** | 0.3853 |

### Discussion

- **Head-only training** benefits from pretrained ImageNet features and achieves reasonable accuracy and AUROC, but remains noticeably overconfident (high ECE).
- **Finetune-all without augmentation** underperforms, suggesting overfitting and instability when all weights are updated without regularization.
- **Finetune-all with AugA** substantially improves accuracy and AUROC, showing that light augmentation is crucial when fully fine-tuning on a small medical dataset.

---

## Augmentation Ablation

| Configuration | Accuracy | AUROC | ECE | Notes |
|---------------|----------|-------|-----|------|
| No augmentation | 0.6218 | 0.6494 | 0.1127 | Overfitting, poor generalization |
| AugA (flip + rotation) | **0.8558** | **0.9703** | 0.3853 | Strong performance, overconfident |

**Observation:**  
Augmentation dramatically improves discriminative performance but increases miscalibration, likely due to the model becoming more confident under expanded data variability.

---

## Learning Curve Interpretation

- **Head-only:** Training and validation curves improve steadily and remain close, indicating stable learning but limited capacity.
- **Finetune-all (basic):** Training accuracy increases rapidly while validation performance stagnates, indicating overfitting.
- **Finetune-all + AugA:** Training and validation curves track closely at high accuracy, suggesting improved generalization.

Augmentation clearly stabilizes finetuning and prevents early overfitting.

---

## Takeaways & Recommended Baseline

- Full finetuning **requires augmentation** to be effective on PneumoniaMNIST.
- The **ResNet18 finetune-all + AugA** configuration provides the best balance of accuracy and AUROC.
- Despite higher ECE, this configuration is the strongest supervised baseline and will be used going into **Week 4 (SSL pretraining)**.

**Chosen supervised baseline:**  
**ResNet18 finetune-all with AugA**

