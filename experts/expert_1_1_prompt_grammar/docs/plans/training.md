# Training
- Use class-balanced data.
    - Ensure equal numbers of positive and negative examples in each batch.
    - Shuffle data before each epoch to prevent ordering bias.
- Monitor metrics on validation set.
    - Track loss, accuracy, precision, recall, and F1-score after each epoch.
    - Save the best model checkpoint based on validation F1-score.
- Analyze confidence score distributions to set a threshold for acceptance/rejection.
    - Plot histograms of confidence scores for both classes on the validation set.
    - Select a threshold that balances false positives and false negatives as needed for downstream use. 