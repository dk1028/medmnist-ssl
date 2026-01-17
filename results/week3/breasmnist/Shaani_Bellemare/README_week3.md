Week 3 Results - breastMNIST
This week goal is to fine the best configuration that will serve as a reference when doing self supervised learning. It builds on week 2 work by comparing our resnet model tuning only the head to Resnet by tuning the whole thing 

1. Setup recap and transforms used
I use data augmentation trhough flipping and rotation it should help with regularization and  since our dataset is really small.

2. Head-only vs finetune-all mini-table + discussion:
Like week 2, I did a small grid search for ResNet when fine tuning the whole the top 3 best hyperparameters are:
ResNet fine tune all: 
LR      WD       Val_AUROC


I need to compare both with the same hyperparams as above so we can;t use week 2 results anymore
ResNet18_HeadOnly     0.  0.  0.
ResNet18_all          0.  0.  0.

3. Augmentation ablation table + commentary


4. Learning curve interpretation


5. Takeaways: which config becomes your reference supervised baseline
ResNet18_all 