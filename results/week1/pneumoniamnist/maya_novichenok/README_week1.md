# Week 1

Dataset: pneumoniamnist
Model: smallcnn
Config: epochs=5, batch_size=128, seed=42

**Issue & Fix**

RuntimeError: 0D or 1D target tensor expected, multi-target not supported

This happened because the dataset labels had shape **[batch_size, 1]**,  
but **CrossEntropyLoss** requires a 1-D tensor of shape **[batch_size]**.

### **Fix**
I resolved this by **squeezing the label dimension** before computing loss.

**Results of TEST:**

Acc:0.0.6218
AUROC:0.6494
ECE:0.11093095861948454

