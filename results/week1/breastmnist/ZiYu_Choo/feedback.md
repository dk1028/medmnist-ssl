Thanks for submitting your Week 1 work. The results and plots are a good start, and it is nice that you compared performance across different epoch settings. Your reported test accuracy, AUROC, and ECE also show that the training pipeline is working.

However, before I can count this as your Week 1 submission, there are a few important consistency issues to fix:

# Dataset mismatch
You are assigned to PneumoniaMNIST, but your config and README both indicate BreastMNIST (DATASET_KEY = 'breastmnist').
Please rerun the Week 1 task on PneumoniaMNIST and update all files accordingly.

# Fine-tuning mismatch
In the README, you wrote “Head Fine-tuning”, but your config shows FINETUNE = 'all', which suggests full fine-tuning.
Please make sure the README matches the actual experiment setting.

# Calibration interpretation
Your ECE is about 0.093, which is reasonable, but I would avoid stating too strongly that the model is “well calibrated” just from that alone.
It would be better to say that the model shows fairly good calibration overall, while briefly commenting on whether the reliability curve is above or below the diagonal in certain confidence ranges.

# Documentation / file organization
Please make sure README_Week1.md and test_metrics.json are clearly separated and consistently named. Right now the contents are a bit confusing.

Once you rerun the experiment on PneumoniaMNIST and fix the documentation inconsistencies, this will be much stronger. Good start overall.
