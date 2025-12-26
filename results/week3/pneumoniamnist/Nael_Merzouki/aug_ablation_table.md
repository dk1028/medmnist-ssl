| Model / Variant            | Finetune | Train Aug   | Test Acc | Test AUROC | Notes               |
|--------------------------- |----------|------------ |---------:|----------: |---------------------|
| ResNet-18 (head-only)      | head     | basic       |   0.76   |    0.84    | Week 2 baseline     |
| ResNet-18 (all, basic)     | all      | basic       |   0.87   |    0.95    | mild overfitting    |
| ResNet-18 (all, AugA)      | all      | flip+rot    |   NULL   |    NULL    | best test AUROC     |
| ResNet-18 (all, AugB)      | all      | flip+rot+jit|   0.89   |    0.98    | similar, more noisy |