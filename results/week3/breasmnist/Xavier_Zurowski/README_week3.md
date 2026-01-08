# **Dataset Recap:**
* The used dataset is BreastMNIST, it contains 780 samples with the following train/val/test splits: 546 / 78 / 156 (Notably smaller than PneumoniaMNIST with 5,856 samples)
* The data set is imbalanced between malignantand benign cases; each split having close to a 1:3 ratio of malignant to benign images.
* Transforms used prior to week3 were T.Resize(64,64) and T.ToTensor() (Note: that Normalized pixel values for resnet were used in w2: (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))

# **Setup Recap**
* Updated to torchvision.transforms.v2 to access v2.GaussianNoise; replaced T.ToTensor() with v2.ToImage() and v2.ToDtype(torch.float32, scale=True); and changed T.Resize((64,64)) to v2.Resize(224) to match the input dimensions expected by ResNet.
* Used v2.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x) to repeat grayscale channel to RGB
* Maintained week2 normalized pixel values (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

# **Head-Only vs Finetune-All**
| Model Variant            | Finetune | Train Acc | Val Acc | Test Acc | Test AUROC | Test ECE | Notes                           |
|--------------------------|----------|-----------|---------|----------|------------|----------|--------------------------------|
| ResNet-18 (head-only)    | head     | 0.7399    | 0.7435 | 0.7243   | 0.5242     | 0.0977   | Frozen backbone limits learning|
| ResNet-18 (all, basic)   | all     | 1.0000    | 0.9102  | 0.8461   | 0.9108    | 0.3204   | Mild overfitting, strong adaptation|

* Head-only training performs poorly, with AUROC near random and a confusion matrix predicting almost exclusively the majority class.
* Allowing Finetune-all substantially improves performance. AUROC jumps to 0.91 demonstrating that the model can now differentiate classes.
* Training accuracy reaches 100%, indicating mild overfitting, though test accuracy does remain relatively high at 84%.

# **Augmentation Ablation**
* **AugA:** Used transforms v2.RandomHorizontalFlip(p=0.5) and v2.RandomRotation(degrees=10). These geometric augmentations slightly reduce test accuracy (roughly 3%) but improve calibration as ECE drops to 0.05. AUROC remains comparable, suggesting model ranking remains strong. Train accuracy alse decreases slighly, indicating a reduction in overfitting.
* **AugB:** Used transforms in AugA as well as v2.ColorJitter(contrast=0.1). Adding color jitter harms performance, likely because low contrast lesions are removed from images. Test accuracy and AUROC drop, the acocompanying rise in ECE shows wekened calibration.
* **AugC** Used transforms in AugA as well as v2.GaussianNoise(sigma=0.01). 

| Model Variant          | Finetune | Train Aug                | Test Acc | Test AUROC | Test ECE | Notes                          |
|------------------------|----------|--------------------------|----------|------------|----------|--------------------------------|
| ResNet-18 (all, basic) | all      | none                     | 0.8461   | 0.9108     | 0.3204    | Reference point, mild overfitting                     |
| ResNet-18 (all, AugA)  | all      | flip + rotation          | 0.8974   | 0.9294      | 0.2995   | Improved test acc, slightly higher AUROC, improved calibration        |
| ResNet-18 (all, AugB)  | all      | flip + rotation + jitter | 0.8589  | 0.9275      | 0.2825    | Improved test acc, slightly higher AUROC, improved calibration     |
| ResNet-18 (all, AugC)  | all      | flip + rotation + Gaussian noise | 0.8397   | 0.8759    | 0.2814  | Lowest test acc and AUROC, best calbiration |

**Conclusion:**
Finetune-all with basic augmentations achieves the highest test accuracy. AugA improves calibration slightly at the price of a small accuracy drop. AugB is harmful for all three metrics. The data suggests that geometric augmentations can help increase accuracy, whereas the contrast augmentations do not. Next steps: Gaussian noise augmentations.

# **Learning Curve Interpretation** 
* **Head_only:** Train loss decreases from ~0.65 to ~0.54 while validation loss stays high at 0.63, indicating limited learning. Train accuracy rises slightly, but validation accuracy is flat at ~0.73 across all epochs suggesting capacity-limited underfitting, and that the model saturates pretty well immediately.
* **Finetune_all:** Train loss drops rapidly to near 0 and train accuracy reaches ~1 by epoch 7. Validation loss spikes severely early at epoch 3 before collapsing, while validation accuracy peaks at ~0.96. Overall suggesting high capacity with unstable early optimization and mild late overfitting.
* **AugA:** Train loss decreases steadily and validation loss closely follows after epoch 4. Train accuracy reaches ~0.95 while validation accuracy increases smoothly to ≈0.90. Data augmentation slows convergence but yields the best generalization with minimal overfitting.

#**Takeaways for SSL (Weeks 4–6)**
Reference Setup: resnet18-all-basic has the best test accuracy and strong AUROC. This will serve as the supervised baseline for SSL experiments. I will be testing some other augmentations in the coming days, readme will be updated if I find anything better.

