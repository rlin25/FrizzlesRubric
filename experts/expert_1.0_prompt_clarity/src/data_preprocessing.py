import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Load dataset from CSV into Hugging Face Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df.reset_index(drop=True))

# Tokenize text column
def tokenize_data(dataset, tokenizer):
    def tokenize_function(examples):
        texts = [str(t) for t in examples["text"]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256
        )

    # Filter to only rows with valid, non-empty text
    dataset = dataset.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 0)

    # Save labels *after* filtering
    labels = dataset["label"]

    # Add 'raw_text' column for the original text before tokenization
    dataset = dataset.add_column("raw_text", dataset["text"])

    # Tokenize and drop non-token fields
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Re-add the labels and raw_text column after tokenization
    if "labels" not in tokenized.column_names:
        tokenized = tokenized.add_column("labels", labels)
    tokenized = tokenized.add_column("raw_text", dataset["raw_text"])

    return tokenized

# Preprocess and split data
def preprocess_data(data, tokenizer_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Handle file path or DataFrame input
    if isinstance(data, str):  # If data is a file path
        dataset = load_data(data)
    elif isinstance(data, pd.DataFrame):  # If data is a pandas DataFrame
        data = data.dropna(subset=["text", "label"]).copy()
        data["text"] = data["text"].astype(str)
        data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(int)
        dataset = Dataset.from_pandas(data.reset_index(drop=True))
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame.")

    # Tokenize the dataset
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    # Set format to PyTorch tensors for model training
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    # Return a train-test split (80/20)
    return tokenized_dataset.train_test_split(test_size=0.2)

# Standalone usage
if __name__ == "__main__":
    dataset = preprocess_data("data/clarity_merged.csv")
    print(dataset)
