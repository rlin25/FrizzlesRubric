# Training Pipeline

## Data Loading
```python
class GranularityDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list, labels: list, tokenizer: DistilBertTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = 512
```

## Training Configuration
```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "num_epochs": 5,
    "early_stopping_patience": 3
}
```

## Cross-Validation
- K-fold cross-validation (k=5)
- Stratified sampling
- Shuffle before splitting
- Random seed: 42

## Early Stopping
```python
def early_stopping(patience: int = 3):
    best_val_score = float('-inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        val_score = evaluate_validation()
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            save_best_model()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
```

## Model Calibration
```python
def calibrate_model(val_data):
    predictions = []
    true_labels = []
    
    for prompt, label in val_data:
        pred, conf = model.predict(prompt)
        predictions.append(conf)
        true_labels.append(label)
    
    return calibration_curve(true_labels, predictions)
```

## Training Loop
1. Load and preprocess data
2. Split into train/validation sets
3. Initialize model and optimizer
4. Train with early stopping
5. Calibrate model
6. Save best model

## Evaluation Metrics
- Binary accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Calibration error

## Logging
- Training loss
- Validation metrics
- Early stopping events
- Calibration results
- Model checkpoints 