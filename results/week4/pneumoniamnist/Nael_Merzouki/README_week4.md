# Week 4 — SSL Training Summary (PneumoniaMNIST)

## 1. Setup

| Parameter | Value |
|-----------|-------|
| **Method** | SimCLR-lite (NT-Xent contrastive loss) |
| **Backbone** | ResNet-18 (ImageNet-pretrained initialization) |
| **Encoder Output Dim** | 512 |
| **Projection Head** | 3-layer MLP (512 → 512 → 128) |
| **Loss Function** | Normalized Temperature-scaled Cross-Entropy (NT-Xent) |
| **Temperature (τ)** | 0.2 |
| **Optimizer** | AdamW |
| **Learning Rate** | 0.001 |
| **Weight Decay** | 0.0001 |
| **Batch Size** | 128 |
| **Epochs** | 30 |
| **Seed** | 42 |
| **Device** | CUDA (GPU) |

### Augmentations (Applied to Both Views)
- RandomResizedCrop (size=64, scale=[0.7, 1.0])
- RandomHorizontalFlip (p=0.5)
- RandomRotation (degrees=10)
- ColorJitter (brightness=0.1, contrast=0.1)
- Normalization (mean=0.5, std=0.5)

## 2. Training Behavior

**Loss Trajectory:**
The loss curve exhibits a smooth, monotonic decline across 30 epochs with no apparent instability. Starting from an initial NT-Xent loss of approximately 3.17, the loss decreases steadily and flattens after epoch 20, indicating convergence toward a stable representation.

**Convergence Pattern:**
By epoch 5, loss drops to ~1.70 (46% reduction). The steepest improvements occur in the first 10 epochs. After epoch 20, the loss plateau widens, suggesting diminishing returns with the final loss (epoch 30) around 1.27. This implies the encoder has captured the primary structure in the unlabeled data.

## 3. Quantitative Results

| Metric | Value |
|--------|-------|
| **Initial Loss (Epoch 1)** | 3.1668 |
| **Final Loss (Epoch 30)** | 1.2654 |
| **Absolute Loss Reduction** | 1.9015 |
| **Relative Loss Decrease** | 60.0% |
| **Loss @ Epoch 5** | ~1.6976 |
| **Loss @ Epoch 10** | ~1.4970 |
| **Loss @ Epoch 15** | ~1.3941 |
| **Loss @ Epoch 20** | ~1.3190 |

**Interpretation:**
A 60% loss reduction over 30 epochs is substantial and indicates strong optimization. The NT-Xent loss measures the encoder's ability to pull positive pairs (two augmented views of the same image) closer and push negatives farther in the projection space. The final loss of 1.27 reflects healthy separation between similar and dissimilar instances.

## 4. Observations & Hypotheses

- **Initial Backbone:** Starting from ImageNet-pretrained ResNet-18 may have accelerated convergence. The high initial loss reflects the task mismatch (ImageNet vs. medical images), but the model quickly adapts.

- **Representation Quality & Limitation/Convergence:** The steady loss decrease without divergence suggests the encoder learns robust, stable features. The plateau after epoch 20 is expected. The loss plateau after epoch 20 suggests diminishing returns.

- **Augmentation Effectiveness:** Medical-specific augmentations to represent real-world noise (rotation ±10°, low-intensity color jitter) preserved image integrity while creating meaningful positive pairs.

- **Optimization Stability:** No divergence or gradient instability suggests AdamW with weight decay 0.0001 and the learning rate of 0.001 is stable.

## 5. Limitations

- **No Validation Metric:** Contrastive learning loss alone does not directly measure downstream task performance. Validation metrics (e.g., linear probe accuracy) are needed to assess representation quality.
- **Single Run:** No variance estimates or multiple random seeds tested.
- **Limited Augmentation Analysis:** Effect of individual augmentations (rotation, jitter, crop scale) not isolated.

---

## Metrics

**[SSL] Final Statistics:**
```
  Initial loss: 3.1668
  Final loss: 1.2654
  Loss reduction: 1.9015
  Loss decreased by: 60.0%
```

**[SSL] Training Loss by Epoch:**
```
  Epoch  1: 3.1668
  Epoch  5: ~1.6976
  Epoch 10: ~1.4970
  Epoch 15: ~1.3941
  Epoch 20: ~1.3190
  Epoch 25: ~1.2816
  Epoch 30: 1.2654
```

**Configuration Applied:**
- Method: SimCLR-lite (NT-Xent Loss with τ=0.2)
- Backbone: ResNet-18 (pretrained=True)
- Feat Dim: 512 | Proj Dim: 128
- Batch Size: 128 | Epochs: 30
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Augmentations: RandomResizedCrop, RandomHorizontalFlip, RandomRotation (±10°), ColorJitter
- Seed: 42
