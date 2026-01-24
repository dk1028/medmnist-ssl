 The overall design is on the right track: SimCLR-lite is a good minimal baseline for MedMNIST-scale experiments, your augmentations are mostly medically plausible (good call avoiding vertical flips/large rotations)

 # what;s strong

Clear method choice & motivation: SimCLR-lite (single encoder, no queue/momentum) is appropriate for a small dataset and matches Week 4 goals.
Augmentation reasoning: The listed transforms (crop/flip/±10° rotation) are reasonable invariances for many CXR settings, and you explicitly avoided unrealistic transforms.
Good Week 5 plan: Correctly notes that the projection head is not used downstream.

# Critical issues to fix (required for a complete submission)

Options (pick at least one):
Save only state_dict() (not full checkpoint/optimizer) → much smaller: torch.save(model.state_dict(), ...)
Save encoder in fp16 (as you did in other submissions) or use safetensors
Use Git LFS for .pt files, or upload to a GitHub Release / Google Drive and link it in the README
At minimum: commit a config file + exact training command + environment versions so someone else can rerun it

# Method clarity improvements (important)

ImageNet-pretrained=True needs a short justification
Using an ImageNet-pretrained encoder means this is not “pure SSL from scratch.” It may still be acceptable as a warm-start, but it changes the interpretation of results.
➜ Please add 1–2 lines: why you used pretrained weights and (ideally) whether you plan to also run a random-init SSL baseline for fairness.

Normalization choice vs pretrained backbone
You used mean/std = 0.5. If the backbone is ImageNet-pretrained, it’s usually best to use ImageNet normalization (or clearly justify dataset-specific normalization).
➜ Please clarify this choice and keep it consistent with the pretrained assumption.

Horizontal flip justification for CXR
Flip can be questionable for CXR because it swaps left/right anatomy. Pneumonia labels may be less laterality-dependent, but it still deserves a short justification or an ablation note (“flip on/off”).
