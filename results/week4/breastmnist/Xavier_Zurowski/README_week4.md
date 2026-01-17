# **Method Choice**
For my SSL pipeline, I implemented SimCLR-lite contrastive learning model made up of a ResNet18 encoder using NT-Xent loss on pairs of augmented views of the same image. 
This method was selected because it is conceptually simple, works with a single encoder, and achieves strong results. 
A lite variant was used since the encoder is ResNet18 (smaller than ResNet50, which is used in the original SimCLR paper), keeping training tractable while preserving the contrastive learning objective.

# **Augmentations**
The model makes use of several augmentations to generate augmented pairs of the same image, which the model then trains on:
* Transforms are applied each image: resize(224), Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x) (ResNet expects images with three channels), ToDtype(torch.float32, scale=True)
  
To create pairs of augmented views the following transforms are applied:

* RandomResizedCrop(224, scale=(0.7, 1.0)) (Randomly crops the image, the remaining portion containing between 70% to 100% of the original image)
* RandomHorizontalFlip(p=0.5) (Randomly, with equal probability flips the image left or right (mirror))
* RandomRotation(degrees=10) (randomly rotates the image by a random angle between -10 and 10 degrees)
* GaussianNoise(sigma=0.01) (Adds pixel-wise Gaussian noise) 
* ColorJitter(brightness=0.05, contrast=0.05) (Randomly changes color brightness and contrast with -5% and 5%)
* Images are then normalized to the values expected by ResNet with Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

These augmentations reflect realistic sources of variability in medical imaging such as changes in field of view, patient positioning, acquisition noise, and scanner-dependent intensity differences, while preserving the underlying anatomical structure.
BreastMNIST images exhibit low contrast and fine-grained texture patterns typical of ultrasound imaging, as such various different runs were conducted with varying transformations:
* Run1: GaussianNoise(sigma=0.01) and ColorJitter(brightness=0.05, contrast=0.05)
* Run2: GaussianNoise(sigma=0.01) (No Jitter)
* Run3: GaussianBlur(kernel_size=3, sigma=(0.1, 0.5) and ColorJitter(brightness=0.05, contrast=0.05)
* Run4: GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)) (No Jitter)
* Run5: ColorJitter(brightness=0.05, contrast=0.05) (No Gaussian augmentation)

# **HyperParameter**
Followed the recommended hyperparameter for week4. Used SEED=42, EPOCHS=50, BATCH_SIZE=128, LR=1e-3, WEIGHT_DECAY=1e-4, TAU=0.2, for all runs.
The encoder and projection head are trained jointly from the start. No learning-rate scheduling or explicit freezing are used this week.

# **Training Behavior**
The NT-Xent loss decreases rapidly during the first few epochs (≈3.46 → ≈2.0), followed by a slower, steady decline as training progresses. After approximately 25–30 epochs, improvements become marginal, with the loss converging around 1.15 by epoch 50.
Training is stable throughout, with only minor oscillations in later epochs, indicating no representation collapse and successful contrastive learning.

Across the five runs with different augmentations, the NT-Xent loss steadily decreased over epochs, converging around 1.12–1.25 in the final epochs.
Runs using only Gaussian noise or only ColorJitter reached slightly lower final losses (~1.14–1.15) than combinations of augmentations, suggesting that adding multiple augmentations did not necessarily improve positive pair alignment for BreastMNIST.
Overall, the loss trends indicate stable contrastive learning, with the model successfully grouping augmented views while pushing apart other samples.
From these results, we can infer which augmentations preserve meaningful ultrasound features versus those that may add excessive variability, guiding selection of realistic transforms for SSL. The best run could be chosen by the lowest final loss or by downstream linear evaluation of class separability.

**Note**
*Encoder weights were saved as float16 to allow the file to fall under GitHubs 25mb limit





