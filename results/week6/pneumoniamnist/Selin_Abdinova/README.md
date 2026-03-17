## Week 6 — SSL Fine-Tuning & Calibration (PneumoniaMNIST)

### Setup

- Dataset: PneumoniaMNIST (binary classification: pneumonia vs normal)
- Backbone: ResNet-18
- SSL method: SimCLR-lite (Week 4 pretrained encoder)
- Batch size: 128
- Fine-tuning:
  - Encoder LR: 1e-5
  - Head LR: 1e-3
  - Epochs: 5

---

### Results

| Model | Accuracy | AUROC | ECE |
|------|---------|------|------|
| SSL probe | 0.806 | 0.927 | — |
| SSL finetune | 0.835 | 0.933 | 0.346 |

---

### Calibration Analysis

The reliability diagram shows significant miscalibration:

- **ECE = 0.346**, which is high.
- The model is strongly **overconfident**, especially for predictions with confidence > 0.7.
- The predicted probabilities do not align well with true accuracy.

Despite good classification performance, the model’s confidence estimates are unreliable.

---

### Observations

- SSL representations are effective: even the linear probe achieves strong AUROC (~0.93).
- Fine-tuning improves performance slightly over the probe.
- However, calibration remains poor, indicating that high-confidence predictions should not be fully trusted.

This reflects a common pattern in medical imaging:
→ **high AUROC but poor calibration**

---

### SSL Training Behavior

The SSL loss curve shows a smooth and consistent decrease:

- Loss decreases from ~4.0 → ~1.55 over 10 epochs
- Indicates stable contrastive learning
- Suggests the encoder learned meaningful representations


