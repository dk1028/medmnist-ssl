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
| ResNet-18 (head-only)    | head     | 0.7344    | 0.7307 | 0.7307   | 0.4392     | 0.1487   | Frozen backbone limits learning|
| ResNet-18 (all, basic)   | all     | 1.0000    | 0.9487  | 0.8974   | 0.9281    | 0.3358   | Mild overfitting, strong adaptation|

* Head-only training performs extremely poorly, collapsing to always predicting the majority class, hence an accuracy of 0.7308, a balanced accuracy of 0.5, zero malignant recall, and a confusion matrix with no malignant predictions. Despite reasonable accuracy, the AUROC of 0.439 and PR-AUC of 0.23 indicate that the model fails to rank malignant cases above benign ones and learns no meaningful discriminative features. 
* Allowing Finetune-all substantially improves performance. AUROC jumps to 0.93 demonstrating that the model can now differentiate classes. The model achieves balanced performance, correctly identifying most malignant cases (76%) while maintaining very high benign recall (95%), with a high balanced accuracy (0.85) and an excellent PR-AUC (~0.86), indicating robust discrimination rather than majority-class bias.
* Training accuracy reaches 100%, indicating mild overfitting, though test accuracy does remain relatively high at 84%.

# **Augmentation Ablation**
* **AugA**: Applied v2.RandomHorizontalFlip(p=0.5) and v2.RandomRotation(degrees=10). These geometric augmentations reduce test accuracy slightly compared to All_basic (0.888 → 0.864) and lower AUROC (0.914 → 0.895), while improving calibration modestly (ECE 0.324 → 0.300). Malignant recall decreases (0.767 → 0.667), indicating that these augmentations may slightly hinder minority-class detection despite stabilizing model predictions.
* **AugB**: Applied the AugA transforms plus v2.ColorJitter(contrast=0.1). Adding color jitter slightly decreases test accuracy relative to AugA (0.864 → 0.863) and AUROC (0.895 → 0.906), while further reducing ECE (0.300 → 0.290). Malignant recall remains low (0.657), suggesting that color jitter does not improve minority-class detection and may add variability without boosting overall discrimination.
* **AugC**: Applied the AugA transforms plus v2.GaussianNoise(sigma=0.01). Test accuracy drops further (0.859), and AUROC declines to 0.895, though calibration remains reasonable (ECE 0.296). Malignant recall stays low (0.657), indicating that adding noise slightly stabilizes confidence but reduces predictive performance for both overall accuracy and minority-class detection.

Table shows the Average value ± standard devation for each value across 5 seeds (42, 46, 5, 133, 67)

| Model variant     | test_acc      | test_auroc   | test_ece     | balanced acc | malignant recall | pr_auc      |
|------------|---------------|--------------|--------------|--------------|-----------------|------------|
| All_basic  | 0.888 ± 0.013 | 0.914 ± 0.018 | 0.324 ± 0.010 | 0.850 ± 0.012 | 0.767 ± 0.026   | 0.818 ± 0.022 |
| All_AugA   | 0.864 ± 0.010 | 0.895 ± 0.010 | 0.300 ± 0.018 | 0.802 ± 0.040 | 0.667 ± 0.108   | 0.789 ± 0.031 |
| All_AugB   | 0.863 ± 0.025 | 0.906 ± 0.012 | 0.290 ± 0.018 | 0.798 ± 0.041 | 0.657 ± 0.084   | 0.795 ± 0.031 |
| All_AugC   | 0.859 ± 0.024 | 0.895 ± 0.019 | 0.296 ± 0.017 | 0.795 ± 0.037 | 0.657 ± 0.069   | 0.794 ± 0.041 |

**Conclusion:**
All_basic is the best method because it maximizes balanced accuracy, malignant recall, and PR-AUC, outperforming all augmentation strategies while maintaining low variability.

# **Learning Curve Interpretation** 
* **Head_only:** Train loss steadily decreases from ~0.62 to ~0.53, while validation loss fluctuates and stays higher ~0.57, indicating only modest generalization gains. Train accuracy improves slightly to ~0.74, but validation accuracy remains nearly flat around ~0.73–0.74 across epochs. This suggests the head-only model is capacity-limited and underfitting and that the model saturates pretty well immediately.
* **Finetune_all:** Train loss drops rapidly to near 0 and train accuracy reaches ~1 by epoch 7. Validation loss spikes severely early at epoch 3 before collapsing, while validation accuracy peaks at ~0.96. Overall suggesting high capacity with unstable early optimization and mild late overfitting.
* **AugA:** Train loss decreases steadily, and after early instability (validation loss spike at epoch 2), validation loss closely tracks train loss from ~epoch 4 onward. Train accuracy rises rapidly ~0.95, while validation accuracy increases steadily ~0.92 with a small, stable gap. This indicates strong learning with good generalization, augmentation slows early convergence and causes initial instability, but ultimately yields a robust model with minimal overfitting and significantly better validation performance than head-only training.


