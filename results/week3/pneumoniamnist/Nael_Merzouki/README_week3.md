# Week 3 - PneumoniaMNIST (resnet18, AdamW, 8 epochs, 128 batch size)

# 1. Setup Recap

* **Dataset:** PneumoniaMNIST (28x28 chest X-ray images of pneumonia cases and normal cases)
* **Class balance:** Significant class imbalance (around 3x more pneumonia cases than normal ones)
*  **Transforms:** Used T.Resize((64, 64)) and T.ToTensor() before week 3.

# 2. Head-only vs Finetune-all 

**Head-only performance:** (frozen backbone, only the head classifier is trainable) Performance was same as week2 (although I think my week2 ece was incorrect since I consistently got much lower results when rerunning): (train & val, test) accuracies at (~0.87, ~0.76), test auroc at ~0.84.

**TO-DO: RECHECK AND FIX HEAD-ONLY METRICS FOR WEEK2**

**Finetune-all:** (all layers trainable) Clear improvement in loss, accuracy, auroc and ece compared to head-only. The additional trainable layers let the model adapt much more to the data than the head-only model: 
* Train & val accuracies went from ~0.87 to ~0.98 (1.00 for train)
* Train & val aurocs went from ~0.92 to ~0.999
* Test accuracy went from ~0.76 to (**TO-DO**)
* Test auroc went from ~0.84 to (**TO-DO**)
* Test ece went from ~(**TO-DO**) to ~0.113

### Comparison table

| Variant                    | Test Acc | Test AUROC | Test ECE   |
|----------------------------|----------|------------|------------|
| ResNet-18 (head-only)      | 0.7628   | 0.8371     | **TO-DO**  |
| ResNet-18 (all, basic)     | **TO-DO**| **TO-DO**  | 0.1129     |

---

# 3. Augmentation Ablation

**TO-DO**: I plan to do additional research on augmentation strategies for medical imaging and expect to finish this section by tomorrow (December 26). So files 'curves_resnet18_all_augA.png' and 'metrics_resnet18_all_augA.jso' are missing from my folder.

# 4. Learning Curve Interpretation

**Head-only:** Showed relatively stable but limited learning. Both training and validation metrics stagnated quickly, suggesting that the frozen backbone features limit the model's learning capabilities for this dataset. Does not seem to be overfitting, but performance was capped.

**Finetune-all:** (basic) Showed strong learning capacity. The model achieved near-perfect training accuracy (~0.98 on test) with excellent AUROC (~0.99), indicating the backbone successfully adapted to pneumonia-specific features. Significant gap between validation ECE (0.0139) and test ECE (0.1129) (shown in the metrics json file) as well as for accuracy suggests some degree of overfitting to the validation set, though overall performance remains excellent.

**Augmentation comments: TO-DO**

## 5. Takeaways for SSL (Weeks 4-6)

**TO-DO**
