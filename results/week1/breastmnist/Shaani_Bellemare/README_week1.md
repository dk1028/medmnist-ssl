Week 1 Results - breastMNIST

Dataset: breastMNIST

Model: ResNet18 with head finetuning 
Epochs: 5
Batch size: 128 

Learning rate: 3e-4  
Weight Decay:  1e-4

Test results
acc=0.7051 
auroc=0.5441
ECE = 0.108
The accuracy and auroc are pretty low (auroc is almost a 50% random guess) it might be a sign of underfitting we might need to tune the hyperparameters. These results will be useful to show improvement of our model later and are a good start for comparison purposes.

one issue + fix: really unrelated but i had a problem with github for a month I just re did the set up from the start to fix it, otherwise i didn't have any big issue with this week