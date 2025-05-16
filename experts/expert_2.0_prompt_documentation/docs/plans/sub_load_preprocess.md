# Expert 2.0: Documentation Evaluation — Data Loading and Preprocessing

## Objective
Load and preprocess the Gemini dataset for training the documentation classifier.

## Data Source
- Location: `data/original_gemini/`
- Format: CSV files
- Content: Natural language prompts with binary labels

## Implementation Steps

### 1. Data Loading
```python
def load_gemini_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from the Gemini dataset directory.
    Args:
        data_dir: Path to directory containing Gemini dataset
    Returns:
        DataFrame containing prompts and labels
    """
    all_data = []
    for file in glob.glob(os.path.join(data_dir, "*.csv")):
        df = pd.read_csv(file)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)
```

### 2. Data Validation
```python
def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate the loaded dataset meets requirements.
    Args:
        df: DataFrame containing prompts and labels
    Returns:
        bool indicating if dataset is valid
    """
    required_columns = ['prompt', 'label']
    if not all(col in df.columns for col in required_columns):
        return False
    
    if not df['label'].isin([0, 1]).all():
        return False
        
    if df['prompt'].isnull().any():
        return False
        
    return True
```

### 3. Text Preprocessing
```python
def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for model input.
    Args:
        text: Raw prompt text
    Returns:
        Preprocessed text
    """
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase
    text = text.lower()
    
    return text
```

### 4. Tokenization
```python
def tokenize_prompts(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Tokenize prompts for model input.
    Args:
        texts: List of preprocessed prompts
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
    Returns:
        Dictionary of tokenized inputs
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
```

### 5. Dataset Creation
```python
class DocumentationDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }
```

### 6. Data Splitting
```python
def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    Args:
        df: Full dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=random_state
    )
    
    return train_df, val_df, test_df
```

## Data Quality Checks
1. Check for duplicate prompts
2. Verify label distribution
3. Analyze prompt lengths
4. Check for missing values
5. Validate text encoding

## Error Handling
```python
class DatasetError(Exception):
    pass

def check_data_quality(df: pd.DataFrame):
    if df.duplicated().any():
        raise DatasetError("Dataset contains duplicate entries")
    
    if df['label'].value_counts().min() < 10:
        raise DatasetError("Dataset is too imbalanced")
        
    if df['prompt'].str.len().max() > 10000:
        raise DatasetError("Dataset contains extremely long prompts")
```

## Performance Optimization
1. Use pandas for efficient data loading
2. Implement batch processing
3. Cache preprocessed data
4. Use multiprocessing for text preprocessing
5. Optimize memory usage

## Testing Requirements
1. Unit tests for data loading
2. Unit tests for preprocessing
3. Unit tests for tokenization
4. Integration tests for dataset creation
5. Performance tests for data processing 