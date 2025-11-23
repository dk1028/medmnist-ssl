Good baseline and clear debugging. You correctly diagnosed the (B,1) → (B,) label shape issue and fixed it. Your metrics show decent discrimination (AUROC ≈ 0.83) with over-confident probabilities (ECE ≈ 0.24).

# Correction 
Make sure you actually reload the best‐val‐AUROC weights before running the test. (It’s easy to accidentally test the last epoch.) (if it looks okay you can ignore this)

