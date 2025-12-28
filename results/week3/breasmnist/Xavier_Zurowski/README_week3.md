# **Dataset Recap:**
* The used dataset is BreastMNIST, it contains 780 samples with the following train/val/test splits: 546 / 78 / 156 (Notably smaller than PneumoniaMNIST with 5,856 samples)
* The data set is imbalanced between malignantand benign cases; each split having close to a 1:3 ratio of malignant to benign images.
* Transforms used prior to week3 were T.Resize(64,64) and T.ToTensor() (Note: that Normalized pixel values for resnet were used in w2: (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))

# **Head-Only vs Finetune-All**
| Model Variant            | Finetune | Train Acc | Val Acc | Test Acc | Test AUROC | Test ECE | Notes                           |
|--------------------------|----------|-----------|---------|----------|------------|----------|--------------------------------|
| ResNet-18 (head-only)    | head     | 0.7344    | 0.7308  | 0.7308   | 0.4392     | 0.1487   | Frozen backbone limits learning|
| ResNet-18 (all, basic)   | all     | 1.0000    | 0.9487  | 0.9103   | 0.9211     | 0.0687   | Mild overfitting, strong adaptation|

* Head-only training performs poorly, with AUROC near random and a confusion matrix predicting almost exclusively the majority class.
* Allowing Finetune-all substantially improves performance. AUROC jumps to 0.92 demonstrating that the model can now differentiate classes.
* Training accuracy reaches 100%, indicating mild overfitting, though test accuracy does remain high at 91%.

# **Augmentation Ablation**
* **AugA:** Used transforms T.RandomHorizontalFlip(p=0.5) and T.RandomRotation(degrees=10). These geometric augmentations slightly reduce test accuracy (roughly 3%) but improve calibration as ECE drops to 0.05. AUROC remains comparable, suggesting model ranking remains strong. Train accuracy alse decreases slighly, indicating a reduction in overfitting.
* **AugB:** Used transforms in AugA as well as T.ColorJitter(contrast=0.1). Adding color jitter harms performance, likely because low contrast lesions are removed from images. Test accuracy and AUROC drop, the acocompanying rise in ECE shows wekened calibration.

| Model Variant          | Finetune | Train Aug                | Test Acc | Test AUROC | Test ECE | Notes                          |
|------------------------|----------|--------------------------|----------|------------|----------|--------------------------------|
| ResNet-18 (all, basic) | all      | none                     | 0.910   | 0.921      | 0.069    | Reference point, mild overfitting                     |
| ResNet-18 (all, AugA)  | all      | flip + rotation          | 0.878   | 0.912      | 0.050    | Slightly lower test acc, improved calibration (ECE)        |
| ResNet-18 (all, AugB)  | all      | flip + rotation + jitter | 0.853   | 0.852      | 0.079    | Lowest test acc, AUROC dropped, minor calibration degradation     |

**Conclusion:**
Finetune-all with basic augmentations achieves the highest test accuracy. AugA improves calibration slightly at the price of a small accuracy drop. AugB is harmful for all three metrics. The data suggests that geometric augmentations can help increase accuracy, whereas the contrast augmentations do not. Next steps: Gaussian noise augmentations.

# **Learning Curve Interpretation** 
* **Head_only:** Train loss decreases from ~0.65 to ~0.54 while validation loss stays high at 0.63, indicating limited learning. Train accuracy rises slightly, but validation accuracy is flat at ~0.73 across all epochs suggesting capacity-limited underfitting, and that the model saturates pretty well immediately.
* **Finetune_all:** Train loss drops rapidly to near 0 and train accuracy reaches ~1 by epoch 7. Validation loss spikes severely early at epoch 3 before collapsing, while validation accuracy peaks at ~0.96. Overall suggesting high capacity with unstable early optimization and mild late overfitting.
* **AugA:** Train loss decreases steadily and validation loss closely follows after epoch 4. Train accuracy reaches ~0.95 while validation accuracy increases smoothly to ≈0.90. Data augmentation slows convergence but yields the best generalization with minimal overfitting.

#**Takeaways for SSL (Weeks 4–6)**
Reference Setup: resnet18-all-basic has the best test accuracy and strong AUROC. This will serve as the supervised baseline for SSL experiments. I will be testing some other augmentations in the coming days, readme will be updated if I find anything better.

