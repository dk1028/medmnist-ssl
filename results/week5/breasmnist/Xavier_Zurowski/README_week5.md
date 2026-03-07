# Week 5 Core Deliverables — Method & Results

## Method Choice  
For week 5, my week4 SSL pretrained pt was used, and I began evaluating the representation quality with linear probing. The SSL encoder was fully frozen, and only a linear classification head was trained. 

In following with week4 baseline, the following data augmentations were applied during SSL pretraining:

```python
ssl_transform = vT.Compose([
    vT.RandomResizedCrop(224, scale=(0.7, 1.0)),
    vT.RandomHorizontalFlip(p=0.5),
    vT.RandomApply([vT.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    vT.RandomRotation(degrees=10),
    vT.RandomGrayscale(p=0.2),
    vT.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    vT.Normalize(mean=[0.485,0.456,0.406],
                 std=[0.229,0.224,0.225]),
```
## Experimental Setup

Linear probes were trained using 5%, 10% and 20% of labeled data, and evaluated on the test split. Performance was measured using accuracy, balanced accuracy (since BreastMNIST is unbalanced), and AUROC. Results were compared against my supervised baseline on 100% of the data. Best threshold was tuned on the validation split of BreastMNIST for balanced accuracy.

Setup for the linear probes:
```
Optimizer: Adam
Learning rate: 1e-3
Epochs: 25
Batch size: 64
Weight decay: None
Loss: CrossEntropyLoss

```

## Results
**Linear Probe Performance**
| Label Fraction | Accuracy | Balanced Accuracy | AUROC | Best Threshold|
| -------------- | -------- | ----------------- | ----- | -------------- |
| 5%             | 0.647   | 0.654            | 0.704 | 0.553 |
| 10%            | 0.609   | 0.672          | 0.736 | 0.698 |
| 20%            | 0.712   | 0.712          | 0.788 | 0.513 |

**Supervised Baselines**
| Model Type                         | Accuracy | Balanced Accuracy | AUROC |
| ---------------------------------- | -------- | ----------------- | ----- |
| Fully supervised (enc + head)      | 0.897    | 0.854             | 0.928 |


## Analysis  
The linear probe results indicate that the SSL-pretrained encoder captures meaningful class-relevant structure even without labeled data. Performance improves consistently as the labeled fraction increases from 5% to 20%, with AUROC rising from 0.704 to 0.788 and balanced accuracy from 0.654 to 0.712. This suggests the encoder's representations become more separable with increased supervision, furthermore it suggests that the learned features carry genuine discriminative signal rather than noise. The threshold values also stabilize closer to 0.5 at 20%, indicating more balanced and confident predictions at higher label fractions.

Despite this, a substantial gap remains between the SSL probe and the fully supervised baseline across all metrics. The supervised model achieves an AUROC of 0.928 compared to 0.788 at the best probe setting — a gap of 0.14 that reflects the fundamental limitation of linear probing: the encoder is frozen and the linear head can only exploit whatever structure already exists in the feature space. Since the SSL encoder was never optimized for this specific task, some discriminative information may be entangled in ways a linear boundary cannot resolve. These results motivate fine-tuning the encoder, which allows the representations themselves to adapt to the classification objective.

