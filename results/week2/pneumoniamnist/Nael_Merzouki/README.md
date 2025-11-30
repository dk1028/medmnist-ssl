# Week 2 — PneumoniaMNIST (resnet18--head, Adam, 8 epochs)

## 1. Dataset Recap

- **Class imbalance:** Checking the counts, PneumoniaMNIST is considerably imbalanced, with more positive cases (pneumonia) than negative ones,
which can make the accuracy & auroc metrics misleading/overinflated. 
- **Class differentiability:** Low resolution of the images (28x28) make them seem blurry and makes it hard to distinguish differences/patterns by human eye but
there seems images labeled "normal" (0) seem to generally have clearer, darker borders around the lungs while images labeled "pneumonia" (1) seem to
have white patches in the center area (where the lung would be).

## 2. Baseline Comparison

- Tried resnet18 and small cnn on epochs between 5-8 as well as Adam optimizer and SGD with momentum. resnet18--head with Adam
seemed like the best consistent performance overall (although smallcnn with SGD reached ece ~0.03 with bad acc and auroc).
smallcnn is probably too small to fully adapt to the dataset and is unable to get an acc of +0.65 and auroc +0.75. Since
resnet18 is larger than smallcnn it can outperform in those metrics.


| Model           | Val Acc | Test Acc | Test AUROC |
| --------------- | ------: | -------: | ---------: |
| smallcnn        |  0.72   |   0.62   |    0.72    | (8 epochs, Adam)
| resnet18 (head) |  0.87   |   0.76   |    0.84    |

## 3. Calibration Snapshot

- The reliability diagram for resnet18 shows overconfidence: accuracy is always considerably lower than confidence.
- The ECE is consistently ~0.25 which pretty high (so model is confidence and accuracy differs significantly).
- The biggest mismatches happen at the higher confidence ends.

Note: I haven’t tried temperature scaling yet b/c of time constraints, but I’m interested to do that in Week 5/6 because it might help with this overconfidence.


## 4. Error Analysis

- Most of the mistakes are normal x-rays that get called pneumonia, which is expected from the class imbalance and overinflated accuracy score.
- Confidence is extremely high when making wrong predictions ~0.99.
- High brightness with hardly noticeable borderlines around the lung. 

---

**Summary:**  
PneumoniaMNIST isn’t balanced, differentiation is difficult with low resolution. resnet18 performs better than smallcnn, 
but still is overconfident with high ece. Next, I want to try some calibration tricks (TS).
