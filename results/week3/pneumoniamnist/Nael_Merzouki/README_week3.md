# Week 3 - PneumoniaMNIST (resnet18, AdamW, 8 epochs, 128 batch size)

# 1. Setup Recap

* **Dataset:** PneumoniaMNIST (28x28 chest X-ray images of pneumonia cases and normal cases)
* **Class balance:** Significant class imbalance (around 3x more pneumonia cases than normal ones)
*  **Basic transforms:** Used T.Resize((64, 64)) and T.ToTensor() before week 3.

# 2. Head-only vs Finetune-all 

**Head-only performance:** (frozen backbone, only the head classifier is trainable) Performance was same as week2: (train & val, test) accuracies at (~0.88, ~0.76), auroc at (~0.93, ~0.84).

**Finetune-all:** (all layers trainable, basic transforms) Clear improvement in loss, accuracy, AUROC and ECE compared to head-only. The additional trainable layers let the model adapt much more to the data than the head-only model: 
* Train & val accuracy: ~0.87 -> ~0.98 (1.00 for train)
* Train & val aurocs: ~0.92 -> ~0.99 (1.00 for train)
* Train & val ece:  ~0.04 -> ~0.01 (0.0004 for train)
* Test accuracy: ~0.76 -> ~0.87
* Test auroc: ~0.84 -> ~0.95
* Test ece: ~0.13 -> ~0.13 (same)

### Comparison table

| Variant                    | Test Acc | Test AUROC | Test ECE   |
|----------------------------|----------|------------|------------|
| ResNet-18 (head-only)      | 0.7628   | 0.8371     | 0.1285     |
| ResNet-18 (all, basic)     | 0.8670   | 0.9546     | 0.1306     |

The model reaches perfect accuracy on train set while it performs at 87% accuracy on test set which indicates mild overfitting. Such decrease in performance are noticeable in AUROC and ECE too, reinforcing this claim. It still significantly outperforms the head-only model on the test set and is thus preferred in practice. Augmentations below show a decrease in overfitting.

---

# 3. Augmentation Ablation

Reference point will be resnet-all-basic metrics. Will only note notable increases/decreases. Refer to all_metrics.txt for detailed metrics.

* **AugA** (Random horizontal flips (p=0.5) + Random rotation (d=15): Slight decrease in acc, AUROC and increase in train loss, ECE while test loss decreased and test acc, AUROC increased by ~2%.
* **AugB** (AugA + Color jitter (brightness, contrast = 0.2)): Slight increase in train loss, ECE from AugA.
* **AugC** (AugA + Normalization (same as ImageNet)): Slight decrease in train loss, ECE and increase in train acc from AugB. Lower train ECE and loss than AugA. Similar test metrics to AugA.
* **AugD** (AugB + Normalization (same as ImageNet)): Best test metrics of all augmentations. Similar train metrics to AugC.

**Best Configuration**: AugD seems to be the best configuration. Although learning curves of AugC might seem more appealing, this is not reflected in the evaluation metrics. It gets the best test metrics (especially accuracy) by considerable margin (which is what matters in practice) and decreases model performance on training set, making overall performance more consistent and decreasing level of overfitting.

| Model / Variant            | Finetune | Train Aug   | Test Acc | Test AUROC | Notes               |
|--------------------------- |----------|------------ |---------:|----------: |---------------------|
| ResNet-18 (head-only)      | head     | basic       |   0.76   |    0.84    | Week 2 baseline     |
| ResNet-18 (all, basic)     | all      | basic       |   0.87   |    0.95    | mild overfitting    |
| ResNet-18 (all, AugA)      | all      | flip+rot    |   0.87   |    0.97    | better test auroc     |
| ResNet-18 (all, AugD)      | all      | flip+rot+jit+norm|   0.9   |    0.98    | best test metrics, more noise |

# 4. Learning Curve Interpretation

We will not comment too much on AugB and AugC curves since they are not too interesting.

**Head-only:** Showed relatively stable but limited learning. Both training and validation metrics stagnated quickly, suggesting that the frozen backbone features limit the model's learning capabilities for this dataset. Similar growth for both train and val metrics. Does not seem to be overfitting, but performance was capped.

**Finetune-all:** (basic) Showed strong learning capacity. The model achieved near-perfect train and val accuracy with excellent AUROC (~0.99), implying the backbone successfully adapted to the training and val set. Train loss and ECE converged to zero while val did not, suggesting mild level of overfitting to the train set, though overall performance remains relatively high. The spikes and difference in smoothness of the growth curves reinforce overfitting claim (train curves remain smooth, val curves contain spikes). Same trends happen with AugB, AugA (less severely), AugC (even less severely).

**Finetune-all + augD**: Clear smoothness of the validation growth curves with much less and smaller spikes. Fast learning rate and better convergence for loss, accuracy and ECE. Overfitting here is much less defined. 

Augmentation D seems to have heavily stabilized learning rate and decrease the level of overfitting. From AugC and AugD curves, we notice the effect of normalization specifically on smoothing out the growth curves.

## 5. Takeaways for SSL (Weeks 4-6)

**Reference Setup**: I will use 'resnet18-all-AugD' as the reference for the next weeks. It achieves the best test results and shows stable training with reasonable generalization.

**Surprises**: Color jitter transformation seems to generally worsen the smoothness of the learning curves (augA -> augB, and augC -> augD for loss & acc) but overfitting levels do not seem to reflect that. Perhaps the smoothness of the learning curves is not enough to diagnose overfitting or maybe this is just a special case?

Note: I think my week2 ECE was incorrect since I consistently got much lower results when rerunning with torchmetrics.BinaryCalibrationError().
