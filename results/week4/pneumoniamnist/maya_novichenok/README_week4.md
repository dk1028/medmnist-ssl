Note: I wasn't able to commit my code or the .pt files as they were too large

Dataset: PneumoniaMNIST
Backbone: ResNet-18 (ImageNet-pretrained = True)
Method: SimCLR-lite (NT-Xent)

Method choice

SimCLR-lite was used as a minimal and reproducible contrastive self-supervised learning baseline. It relies on a single encoder without a momentum encoder or negative queue, making it well suited for small-scale MedMNIST experiments while still learning useful representations.

Augmentations

Two strong but medically plausible augmented views are generated per image using:

Resize to 224 × 224 followed by RandomResizedCrop (scale 0.7–1.0)

RandomHorizontalFlip (p = 0.5)

RandomRotation (±10°)

Mild ColorJitter (brightness = 0.1, contrast = 0.1)

Grayscale converted to 3 channels

Normalization with mean = [0.5, 0.5, 0.5] and std = [0.5, 0.5, 0.5]

Vertical flips and large rotations were avoided to preserve anatomical plausibility in chest X-rays.

Hyperparameters

Projection head: 2-layer MLP (512 → 128 → 128)

Batch size: 128

Epochs: 30

Optimizer: AdamW

Learning rate: 1e-3

Weight decay: 1e-4

Temperature (τ): 0.2

Seed: 42

Training behavior (loss)

The contrastive NT-Xent loss decreases rapidly during the early epochs and then gradually plateaus, indicating stable optimization and successful representation learning without collapse. The full loss curve is saved as ssl_loss_curve.png.

Notes for Week 5

The pretrained encoder is saved as ssl_encoder.pt.

The projection head (ssl_proj_head.pt) is not used for downstream tasks.

In Week 5, the encoder will be reused for supervised finetuning or linear evaluation, with the projection head discarded or reinitialized.
