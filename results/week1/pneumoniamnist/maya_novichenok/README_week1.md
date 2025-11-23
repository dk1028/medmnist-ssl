# Week 1

Dataset: pneumoniamnist
Model: resnet18 --finetune head
Config: epochs=5, batch_size=128, seed=42

**Issue & Fix**

Issue:
During training, the run failed with
"RuntimeError: 0D or 1D target tensor expected, multi-target not supported."
This happened because the dataset labels had shape [batch_size, 1], while CrossEntropyLoss expects a 1-D tensor of class indices [batch_size].

Solution:
I fixed this by squeezing the label dimension before computing the loss:

y = y.squeeze(1).long()
loss = criterion(logits, y)

This converts [B, 1] → [B] and allows CrossEntropyLoss to work correctly.

**Results of TEST:**

Acc:0.7628
AUROC:0.8252
ECE:0.2389

Note: The model reached moderate performance (Acc ≈ 0.76, AUROC ≈ 0.83). AUROC > Acc indicates that the classifier ranks positive samples fairly well, but threshold calibration could improve accuracy.
