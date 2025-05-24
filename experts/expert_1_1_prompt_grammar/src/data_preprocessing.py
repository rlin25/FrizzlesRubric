import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text):
    """Basic text preprocessing."""
    # Convert to string if not already
    text = str(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_data(data_dir, tokenizer_name="distilbert-base-uncased"):
    """Preprocess the dataset and create train/test splits."""
    logger.info("Loading and preprocessing data...")
    
    # Load the dataset
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_data = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Preprocess texts
    train_data['sentence'] = train_data['sentence'].apply(preprocess_text)
    val_data['sentence'] = val_data['sentence'].apply(preprocess_text)
    test_data['sentence'] = test_data['sentence'].apply(preprocess_text)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    logger.info("Data preprocessing completed.")
    
    return {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    } 