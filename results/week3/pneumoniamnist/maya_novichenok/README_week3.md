1.Setup recap and transforms used.

All experiments use a ResNet-18 backbone pretrained on ImageNet. Models were trained using cross-entropy loss with identical optimizer settings across regimes to isolate the effects of trainability and augmentation.

Training regimes: head-only, finetune-all (basic), and finetune-all with Augmentation (+AugA)

Data augmentation (AugA) consisted of RandomHorizontalFlip(p=0.5) and RandomRotaion(degrees=10); Vertical rotation was avoided as chest X-rays would be completely illegible/upside down. Large rotations were avoided for the same reason.

2.Head-only vs finetune-all mini-table + discussion.

REGIME;        TRAIN ACC;  VAL ACC;  VAL LOSS BEHAVIOUR
Head-only:     ~92.5%;     ~91.2%;    Smoothe, stable
Finetune-all;  ~99.0%;     ~96-98%;   Less stable, large oscillations

The head-only model converges smoothly with a small train–validation gap, indicating stable training and limited overfitting, though with lower overall performance. In contrast, finetuning all layers without augmentation achieves near-perfect training accuracy but shows unstable validation behavior, consistent with overfitting on a small medical dataset.


3.Augmentation ablation table + commentary.

REGIME;               TRAIN ACC;  VAL ACC;  STABILITY
Finetune-all (basic)  ~99%;       ~96-97%   Unstable
Finetune-all (+AugA)  ~97.8%;     ~98.2%    Improved stability from basic

Light augmentations stabilize validation performance: training accuracy decreases slightly due to regularization, but validation accuracy improves and avoids the severe loss spikes seen in unaugmented finetuning, indicating better generalization.

4.Learning curve interpretation.

Head-only:

Underfits slightly but generalizes reliably
Smooth loss decay, small train–val gap
Strong candidate for controlled comparisons

Finetune-all (no aug):
Fast convergence and very low training loss
Clear signs of overfitting and validation instability
High variance across epochs

Finetune-all + AugA:
Best overall validation accuracy
Reduced overfitting compared to basic finetuning
More robust training dynamics

Overall, training dynamics align with expectations for small medical imaging datasets: capacity increases must be paired with regularization.

5.Takeaways: which config becomes your reference supervised baseline.
ResNet-18, finetune-all, with light medically plausible augmentations (AugA). Finetune-all + light augmentation provides the best tradeoff between capacity and generalization.
