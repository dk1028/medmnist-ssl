Dataset: BreastMNIST (binary classification: benign vs malignant)  
Model: ResNet18, fine-tuning head only  
Training: 5 epochs, batch size 128, learning rate 3e-4  

**Results on Test Set:**  
- Accuracy: 0.7051  
- AUROC: 0.5441  
- Expected Calibration Error (ECE): 0.108  

This baseline will serve as a reference for future self-supervised pretraining and label-efficiency experiments.
