# P1 — Temperature Scaling (PneumoniaMNIST)

## Model
SSL fine-tuned ResNet-18

## Fitted Temperature
T = 0.9763

## Results (Validation)

| Metric | Before | After |
|--------|--------|--------|
| ECE    | 0.1786 | 0.1771 |

## Observations
- AUROC remains approximately unchanged (expected — scaling preserves ranking).
- ECE decreases after temperature scaling.
- Reliability diagram shows reduced overconfidence in high-probability bins.
