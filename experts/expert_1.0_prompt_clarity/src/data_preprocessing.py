# src/data_preprocessing.py

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Load dataset from CSV into Hugging Face Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    return dataset

# Tokenize text column
def tokenize_data(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=256  # You can tweak this based on average prompt length
        )
    return dataset.map(tokenize_function, batched=True)

# Preprocess and split data
def preprocess_data(file_path, tokenizer_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = load_data(file_path)
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    # Rename 'label' to 'labels' to match Trainer expectations
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Set only model inputs as tensors — keep 'text' for evaluation/logging
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    # Return train/test split (default 80/20)
    return tokenized_dataset.train_test_split(test_size=0.2)

# Standalone usage
if __name__ == "__main__":
    dataset = preprocess_data("data/prompt_clarity_dataset_clean.csv")
    print(dataset)
