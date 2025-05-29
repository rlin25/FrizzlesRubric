# Model
- **Architecture:** DistilBERT (pretrained, fine-tuned for binary classification)
    - Use HuggingFace Transformers library for model loading and training.
    - Add a classification head (dense layer + softmax/sigmoid) for binary output.
    - Freeze base layers for initial epochs if needed, then unfreeze for full fine-tuning.
- **Metrics:** Track accuracy, precision, recall, F1-score, and confusion matrix during evaluation (not output by API).
    - Compute metrics on validation and test sets after each training run.
    - Log metrics to console and optionally to a file for later analysis.
    - Confusion matrix is used to analyze types of errors (false positives/negatives). 