# Experiment 1: Linear Probing of StreetCLIP Embeddings

## Objective
Determine how much geographic information is encoded in PLONK's frozen StreetCLIP backbone.

## Method
1. Extract 1024-dim feature vectors from frozen StreetCLIP for 50,000 OSV-5M test images
2. Train logistic regression classifier: features → country label (210 classes)
3. Evaluate on 20% held-out test set (9,999 images)

## Results

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | **84.97%** |
| **Top-5 Accuracy** | 97.26% |
| Random Baseline | 0.48% |
| Training Samples | 39,994 |
| Test Samples | 9,999 |
| Countries | 210 |

## Key Finding

**Geographic information is strongly encoded in StreetCLIP embeddings before any geolocation-specific training.** A simple linear classifier achieves 85% country accuracy, 178x better than random guessing.

## Interpretation

PLONK's flow matching head operates on features that already contain rich geographic information. The generative component primarily:
1. **Refines** the existing geographic knowledge
2. **Calibrates** uncertainty estimates
3. **Handles** ambiguous cases through probabilistic prediction

Rather than learning geography from scratch, PLONK leverages pre-existing spatial knowledge in StreetCLIP.
