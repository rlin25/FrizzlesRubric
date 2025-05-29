# Training and Evaluation Plan

## Training
- Use binary cross-entropy loss for binary classification.
- Optimizer: AdamW.
- Train for several epochs, monitoring validation loss and metrics.

## Evaluation Metrics
- Track accuracy, precision, recall, and F1 score on the validation set.
- Emphasize recall to ensure most file references are detected.

## Threshold Tuning
- Tune the classification threshold based on validation results to balance recall and precision.

## Model Saving
- Save the best model based on validation F1 or recall. 