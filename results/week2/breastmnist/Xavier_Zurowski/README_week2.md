**Dataset Recap:**

Breastmnist is the smaller dataset of the 2, split values being (train, val, test): 546, 78, 156
Within that the specific label values are (Class 0: Malignant, Class 1: Benign):
| Split | Class 0 | Class 1 | Total |
|-------|---------|---------|-------|
| Train | 147     | 399     | 546   |
| Val   | 21      | 57      | 78    |
| Test  | 42      | 114     | 156   |

The data between all splits is unbalanced, With each split having close to a 1:3 ratio of malignant to benign images.

Visual differences:
* Malignant Cases tended to be darker, with more of the image consisting of black pixels. 
* Dark holes were present in both cases, malignant cases tended to have holes that were less circular, instead featuring jagged edges.
* Malignant samples often featured dark pixels from the bottom of the sample converging to a point in the center of the sample.

**Baseline Comparison:**

Models:
This week I did several runs with a multitude of changes to improve the performance of both models. My baseline run from this week without any changes: 
| Model    | Accuracy | AUROC  |
|----------|----------|--------|
| SmallCNN | 0.7308   | 0.5731 |
| ResNet18 | 0.7051   | 0.5441 |

In following from last week, and confirmed by the confusion matrices of the run, the models were largely fitted to the class imbalance in the data, smallcnn guess benign with 100% frequency

I made several changes to the models:
* Increased Epochs from 5 to 8
* Lowered learaning rate for Resnet18 from 3e-4 to 1e-4
* Normalized pixel values for Resnet18 in following with documentation: (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
* Changed finetune='head' to finetune='all' for resnet18
* Added weights to the criterion of the Smallcnn model in an effort to minimize class imbalance.

Subsequent runs yielded the following stats:
| Model    | Val Accuracy | Test Accuracy | Test AUROC |
|----------|-------------|---------------|------------|
| SmallCNN | 0.7436      | 0.7179        | 0.6051     |
| ResNet18 | 0.8590      | 0.8397        | 0.8761     |

ResNet18 performs significantly better than SmallCNN, showing higher validation and test accuracy and a much stronger AUROC. Since the dataset is already difficult to interpret even by the human eye
likely that ResNet18’s deeper architecture and greater capacity allow it to capture subtle patterns that the leaner SmallCNN cannot.
* SmallCNN appears underconfident: its low AUROC relative to accuracy suggests poorly calibrated probability estimates
* ResNet18 is far better calibrated, with its high AUROC closely matching its strong accuracy, indicating more reliable confidence in its predictions.

**Calibration Snapshot:**

* ResNet18 is generally well-calibrated, with low ECE values (≈0.056–0.081).
* Low-confidence bins have too few samples to be meaningful.
* In mid-confidence regions the model is slightly underconfident, with accuracy exceeding predicted probability.
* High-confidence bins (0.8–1.0), where most predictions lie, align closely with the diagonal, indicating strong calibration where it matters most.

**Error Analysis:**

* Most errors are false negatives, usually lighter malignant images lacking typical malignant texture.
* Rare false positives tend to be unusually dark benign images.
* The model appears sensitive to the overall proportion of light vs. dark pixels.
* Several errors are visually ambiguous even to a human observer, suggesting inherent data difficulty.
* Class imbalance likely contributes to systematic bias in borderline cases.

