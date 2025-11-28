# Week 1

Dataset: pneumoniamnist  
Model: smallcnn  
Config: epochs=5, batch_size=128, seed=42

---

## Issue & Fix

Issue: During training, outputs (best.pt, metrics, reliability plot) did not save automatically.  
Solution: I added manual saving code using the `shutil` library

## Results
- **Final Test Accuracy:** 0.6426  
- **Final AUROC:** 0.7504  
- **Calibration (ECE):** 0.138  

The model achieved moderate performance with AUROC ~0.75, showing reasonable discrimination between pneumonia and normal classes.  
Calibration was acceptable for a simple baseline model, with ECE around 0.138.  
This run provides a clean baseline that I can compare against in future SSL weeks.
