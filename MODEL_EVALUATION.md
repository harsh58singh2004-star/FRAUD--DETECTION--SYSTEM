# Model evaluation (reproducible local run)

Training command: `python train.py`

The supplied synthetic CSV contains 12,000 rows, including 413 positive fraud labels (3.4417%). A stratified 80/20 split is used (9,600 training / 2,400 test rows). SMOTE is placed inside the imbalanced-learn pipeline and therefore is applied only to the training partition during fitting; the held-out test partition remains at its natural class prevalence.

## Current run — random_state 42, threshold 0.50

| Metric | Score |
|---|---:|
| Accuracy | 0.8833 |
| Precision (fraud) | 0.2094 |
| Recall (fraud) | 0.8554 |
| F1 (fraud) | 0.3365 |
| ROC-AUC | 0.9368 |
| Average precision / PR-AUC | 0.4505 |

Confusion matrix: TN=2,049, FP=268, FN=12, TP=71.

Interpretation: at this threshold, the model catches 71 of 83 fraud-labelled test rows, but also flags 268 legitimate rows. This illustrates the precision/recall trade-off and why a fraud workflow needs a review queue rather than automatic punitive decisions. A threshold should be chosen using validation data and the relative costs of missed fraud versus false alerts.

**Critical limitation:** labels in the supplied dataset were generated synthetically from a hidden logistic risk formula. Metrics measure recovery of that synthetic pattern, not generalization to real financial crime. Treat all figures as educational demonstrations, not claims of bank-grade performance.
