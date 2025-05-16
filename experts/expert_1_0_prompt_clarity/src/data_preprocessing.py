import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

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

    # Save labels after filtering
    labels = dataset["label"]

    # Save raw text before tokenization
    dataset = dataset.add_column("raw_text", dataset["text"])

    # Tokenize and drop non-token fields
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Re-add labels and raw_text column after tokenization
    tokenized = tokenized.add_column("labels", labels)
    tokenized = tokenized.add_column("raw_text", dataset["raw_text"])

    return tokenized

# Preprocess and split data
def preprocess_data(data, tokenizer_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Handle file path or DataFrame input
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame.")

    # Drop bad rows
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    # ✅ Remove exact duplicates
    df = df.drop_duplicates(subset=["text"])

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 🚨 Check for overlap (should be zero)
    overlap = set(train_df["text"]).intersection(set(test_df["text"]))
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping prompts!"

    # ✅ Save test set to CSV for evaluation
    train_df.to_csv("data/clarity_train_set.csv", index=False)
    test_df.to_csv("data/clarity_test_set.csv", index=False)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Tokenize separately
    tokenized_train = tokenize_data(train_dataset, tokenizer)
    tokenized_test = tokenize_data(test_dataset, tokenizer)

    # Set format for PyTorch
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return DatasetDict({
        "train": tokenized_train,
        "test": tokenized_test
    })

# Standalone usage
if __name__ == "__main__":
    dataset = preprocess_data("data/clarity_merged.csv")
    print(dataset)