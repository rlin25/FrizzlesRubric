# Training Pseudocode

## GrammarDataset Class

### Initialization
```pseudocode
Class GrammarDataset:
    Constructor(texts, labels, tokenizer, max_length = 512):
        Store texts, labels, tokenizer, and max_length
```

### Dataset Methods
```pseudocode
Method __len__():
    Return number of samples

Method __getitem__(idx):
    Get text and label at index
    Tokenize text with:
        - Add special tokens
        - Max length padding
        - Truncation
    Return dictionary with:
        - input_ids
        - attention_mask
        - labels
```

## GrammarTrainer Class

### Initialization
```pseudocode
Class GrammarTrainer:
    Constructor(model_name = "distilbert-base-uncased", output_dir = "models"):
        Set device (CPU/GPU)
        Load tokenizer and model
        Create output directory
```

### Data Preparation
```pseudocode
Method prepare_data(train_path, val_path, batch_size = 16):
    Load train and validation CSV files
    Create GrammarDataset instances
    Create DataLoaders with:
        - Specified batch size
        - Shuffling for training
    Return train and validation loaders
```

### Training Loop
```pseudocode
Method train(train_loader, val_loader, num_epochs = 3, learning_rate = 2e-5):
    Initialize optimizer (AdamW)
    Create learning rate scheduler
    Initialize best F1 score

    For each epoch:
        # Training phase
        Set model to training mode
        For each batch:
            Move batch to device
            Forward pass
            Calculate loss
            Backward pass
            Update weights
            Step scheduler

        # Validation phase
        Calculate validation metrics
        Log metrics
        If F1 score improved:
            Save model
```

### Evaluation
```pseudocode
Method evaluate(val_loader):
    Set model to evaluation mode
    Initialize prediction and label lists

    For each batch:
        Get model predictions
        Store predictions and true labels

    Calculate metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 score

    Return metrics dictionary
```

### Model Saving
```pseudocode
Method save_model():
    Save model weights
    Save tokenizer
    Log save location
```

## Main Execution Flow
```pseudocode
Create GrammarTrainer instance
Load and prepare data
Train model
Save best model
``` 