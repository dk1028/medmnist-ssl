## P4 — High-Confidence Error Analysis

### Observations

- False positives often show lung patterns that resemble pneumonia but are subtle or ambiguous.
- False negatives tend to have low contrast or faint signs of infection.
- The model appears overconfident in incorrect predictions, consistent with high ECE observed in Week 6.

### Insight

Model errors suggest that:
- The model struggles with borderline cases.
- Confidence scores are not reliable indicators of correctness.
