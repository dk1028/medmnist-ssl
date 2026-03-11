# Week 1 Results - BreastMNIST

---

### Model Configuration
* **Dataset:** `BreastMNIST`
* **Model:** ResNet18 (Head Fine-tuning)
* **Epochs:** 25 *(Increased from 5 to improve fitting and prevent underfitting)*
* **Batch Size:** 128
* **Learning Rate:** $1 \times 10^{-4}$
* **Weight Decay:** $1 \times 10^{-4}$

---

### Test Performance
| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | 0.8652482269503546 |
| **AUROC** | 0.8884670147128618 |
| **ECE** | 0.0929770735984153 |

---

### Analysis & Observations

**Impact of Training Duration:**
I believe that the test accuracy is influenced by epochs, as when I had epochs as 5. Test accurary and AUROC was at 0.7659574468085106 and 0.7712387280493593 respectively. 

**Model Performance:**
Having a AUROC at approximately 0.88 ($>0.5$) suggests that the model performs well in distingushing beween classes. 

**Calibration:**
From my understanding, a low ECE suggests that the model is well calibrated, the model's confidence scores are quite aligned with its accuracy.

**What I Learned:**
For this week's task I learned a lot about how calibration shows the model's "confidence" in classification and the influence of epoch on the model's performance. 