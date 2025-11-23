1. 
- The dataset is fairly imbalanced with a near 3:1 distribution of pneumonia to normal training and validation data and a 3:2 ratio (pneumonia to normal) of test data. 
- There do not seem to be any surprising artifacts in the dataset.
- Some images seem clearly semperable, however much of the data seems like it could be part of either class.

2.
Model:  Val Acc;  Test Acc;  Test AUROC
smallcnn: 0.7424; 0.6234; 0.6252
resnet18(head): 0.8569; 0.7628; 0.8252

The resnet18 model performs better (with higher accuracy). This is likely because the ResNet architecture uses residual skip connections, solving the vanishing gradient problem many CNNs run into. Furthermore, the smallcnn is designed to be fast and efficient meaning it has fewer parameters and layers, which limits its ability to learn the complex features necessary to achieve the higher accuracy of a deeper model like ResNet18 on a challenging dataset.

The model is slightly over-confident.
The accuracy curve sits below the diagonal in most bins → confidence > actual accuracy.

Low-confidence bins (0–0.5):
Very few samples → calibration not meaningful.

Mid/high-confidence bins (0.5–1.0):
Most samples are here, and accuracy is consistently lower than predicted confidence → over-confidence strongest at high confidence.


3.
The model’s ECE after temperature scaling is 0.093, which is higher than before (~0.066), indicating that TS worsened calibration in this case. In the reliability diagram, the accuracy curve still sits slightly below the diagonal in most bins, meaning the model remains mildly over-confident, especially in the high-confidence region (0.8–1.0). Temperature scaling shifted the confidence values but did not fix the miscalibration pattern, and overall discrimination metrics such as AUROC remain unchanged, since TS affects only confidence magnitudes, not ranking.

4.
Nearly all misclassified examples have a true label of 0 (normal) and a predicted label of 1. Borderline intensity and overall image "fogginess" may have led to this prediction, as each image looks visually closer to the other class to me as well.
