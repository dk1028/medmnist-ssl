Thanks for submitting your Week 1 work. The results and plots are a good start, and it is nice that you compared performance across different epoch settings. Your reported test accuracy, AUROC, and ECE also show that the training pipeline is working.

However, before I can count this as your Week 1 submission, there are a few important consistency issues to fix:

# Calibration interpretation
Your ECE is about 0.093, which is reasonable, but I would avoid stating too strongly that the model is “well calibrated” just from that alone.
It would be better to say that the model shows fairly good calibration overall, while briefly commenting on whether the reliability curve is above or below the diagonal in certain confidence ranges.

# Documentation / file organization
Please make sure README_Week1.md and test_metrics.json are clearly separated and consistently named. Right now the contents are a bit confusing.
