| SSL Augmentation | Accuracy | AUROC |
|------------------|---------|------|
| RandomResizedCrop + Flip + Rotation | 0.8894230769230769 | 0.9664584703046242 |

The SSL encoder trained with random crops, horizontal flips, and small rotations
produced useful representations for downstream classification. The linear probe
achieved strong AUROC, indicating that the learned features capture important
patterns in chest X-ray images.
