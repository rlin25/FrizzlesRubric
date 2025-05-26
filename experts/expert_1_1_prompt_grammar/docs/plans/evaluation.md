# Evaluation
- Use test scripts to evaluate the model and API live from the command line.
    - Provide CLI scripts that send example prompts to the API and print results.
    - Include both positive and negative test cases for comprehensive coverage.
    - Scripts should log responses, errors, and latency for each request.
- Use stratified test set for final evaluation.
    - The test set is held out and not used during training or validation.
    - Evaluation metrics (accuracy, precision, recall, F1, confusion matrix) are computed on this set.
    - Results are logged and optionally saved to a report file for review. 