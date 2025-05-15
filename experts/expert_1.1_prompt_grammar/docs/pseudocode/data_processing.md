# Data Processing Pseudocode

## JFLEGProcessor Class

### Initialization
```pseudocode
Class JFLEGProcessor:
    Constructor(output_dir = "processed_data"):
        Create output directory if it doesn't exist
        Initialize data as None
```

### Load Dataset
```pseudocode
Method load_jfleg():
    Load JFLEG dataset using HuggingFace datasets
    Convert train and test splits to pandas DataFrames
    Combine train and test data
    Log number of samples loaded
```

### Prepare Data
```pseudocode
Method prepare_data():
    If no data loaded:
        Raise error

    # Create incorrect samples (class 0)
    Extract original sentences
    Assign label 0

    # Create correct samples (class 1)
    For each original sentence:
        For each reference correction:
            Add to correct samples with label 1

    # Combine and shuffle
    Concatenate incorrect and correct samples
    Shuffle with fixed random seed

    # Split data
    Split 70% for training
    Split remaining 30% into validation (15%) and test (15%)
    
    Log split sizes
    Return train, validation, and test DataFrames
```

### Save Data
```pseudocode
Method save_splits(train_data, val_data, test_data):
    Save each split as CSV file
    Calculate and save dataset statistics:
        - Split sizes
        - Class distributions
    Save statistics as JSON
```

## Main Execution Flow
```pseudocode
Create JFLEGProcessor instance
Load JFLEG dataset
Prepare data splits
Save processed data and statistics
``` 