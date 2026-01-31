# Week 5 Core Deliverables — Method & Results

## Method Choice  
For week 5, my week4 SSL pretrained pt was used, and I began evaluating the represnatation quality with linear probing. The SSL encoder was fully frozen, and only a linear classification head was trained. 

In following with week4 baseline, the following data augmentations were applied during SSL pretraining:

```python
vT.RandomResizedCrop(224, scale=(0.7, 1.0)),
vT.RandomHorizontalFlip(p=0.5),
vT.RandomRotation(degrees=10),
vT.GaussianNoise(sigma=0.01),
vT.ColorJitter(brightness=0.05, contrast=0.05),
vT.Normalize(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]),
```
## Experimental Setup

Linear probes were trained using 5% and 10% of labeled data, and evaluated on the test split. Performance was measured using accuracy, balanced accuracy (since BreasMNIST is unbalanced), and AUROC. Results were compared against two supervised baselines: a frozen-encoder supervised head and a fully supervised model.

## Results
**Linear Probe Performance**
| Label Fraction | Accuracy | Balanced Accuracy | AUROC |
| -------------- | -------- | ----------------- | ----- |
| 5%             | 0.641    | 0.604             | 0.692 |
| 10%            | 0.724    | 0.706             | 0.781 |

**Supervised Baselines**
| Model Type                         | Accuracy | Balanced Accuracy | AUROC |
| ---------------------------------- | -------- | ----------------- | ----- |
| Supervised (head only, frozen enc) | 0.730    | 0.500             | 0.439 |
| Fully supervised (enc + head)      | 0.897    | 0.854             | 0.928 |

## Analysis

## Analysis  
The linear probe significantly outperforms the supervised head-only baseline in AUROC, particularly at 10% labels, indicating that the SSL encoder learns discriminative representations beyond what is achievable with limited supervised training alone. Note that the supervised head-only model collapses to predicting the majority class and does not meaningfully learn feature representation. Hence, comparisons based on accuracy and/or AUROC are difficult to interpret, since the baseline performance reflects the collapse rather than true learning. That being said, the linear probe does not collapse to the majority class, therefore providing a more meaningful measure of the quality of the SSL representations.  

Although fully supervised training achieves the best overall performance, the linear probe results demonstrate strong label efficiency and validate the effectiveness of the SSL pretraining pipeline. The performance gap between 5% and 10% further highlights the benefits of even modest increases in labeled data when using high-quality SSL representations.

