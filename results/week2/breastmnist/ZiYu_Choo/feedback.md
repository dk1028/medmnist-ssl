Thanks for the Week 2 submission. This is a solid comparison overall, and I like that you included multiple evaluation views such as confusion matrices, reliability diagrams, and image galleries rather than only reporting summary metrics.

# A few strengths stand out:

# The comparison between smallCNN and ResNet18 is very clear.

ResNet18 performs substantially better overall, with higher test accuracy (0.844 vs. 0.674), much stronger AUROC (0.885 vs. 0.543), and lower ECE (0.079 vs. 0.157).

The confusion matrices make it clear that ResNet18 is much more balanced across classes.

# A few points to improve in your write-up and figures:

Discuss the class bias of smallCNN more explicitly.
From the confusion matrix, smallCNN predicts class 1 far too often:

TN = 4, FP = 39

FN = 7, TP = 91
This means it has very poor performance on the negative class, even though its recall on the positive class is high. Please mention that the model is strongly biased toward the positive class rather than only saying that it performs worse overall.

Interpret AUROC more precisely.
ResNet18’s AUROC is strong, while smallCNN’s AUROC is only slightly above random guessing. It would be better to explicitly say that smallCNN has weak class discrimination ability.

Strengthen the calibration discussion.
The reliability diagrams and ECE values show that ResNet18 is also better calibrated than smallCNN, not just more accurate. Please make that distinction clearly.

Check the “misclassified image gallery.”
Some images shown in the gallery appear to be correctly classified (for example, cases where True = Pred). Please verify whether the filtering or labeling logic is correct.




Overall, this is a good Week 2 submission with a meaningful model comparison. The main improvements needed are deeper interpretation of the confusion matrices and calibration results, and checking the correctness of the misclassified gallery.
