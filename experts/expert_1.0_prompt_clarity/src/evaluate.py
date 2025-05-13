import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import os

def compute_metrics(predictions, labels):
    """Compute evaluation metrics (accuracy, precision, recall, F1)."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def load_data(file_path):
    """Load dataset from CSV into a Hugging Face Dataset."""
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df.reset_index(drop=True))

def tokenize_data(dataset, tokenizer_name="bert-base-uncased"):
    """Tokenize the text data."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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

    # ✅ Rename 'label' to 'labels'
    if "label" in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")

    # ✅ Remove all columns except 'text' and 'labels'
    columns_to_remove = [col for col in dataset.column_names if col not in ['text', 'labels']]
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)

    # ✅ Set tensor format
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_dataset, tokenizer

def evaluate_model(model_path, test_file, tokenizer_name="bert-base-uncased", device='cuda'):
    """Evaluate the model on the test set and print metrics."""

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load and tokenize the test dataset
    test_dataset = load_data(test_file)
    tokenized_test_dataset, _ = tokenize_data(test_dataset, tokenizer_name)

    # Make predictions
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in tokenized_test_dataset:
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            labels = batch['labels'].unsqueeze(0).to(device)  # ✅ correct field
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).cpu().numpy()
            actual_label = labels.cpu().numpy()
            predictions.extend(predicted_class)
            actual_labels.extend(actual_label)

    # Calculate metrics
    metrics = compute_metrics(predictions, actual_labels)

    print(f"\n📊 Evaluation Metrics for model: {model_path}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset.")
    parser.add_argument("--model_path", required=True, help="Path to the trained model directory.")
    parser.add_argument("--test_file", required=True, help="Path to the test dataset CSV file.")
    parser.add_argument("--tokenizer_name", default="bert-base-uncased", help="Tokenizer name (default: bert-base-uncased).")
    parser.add_argument("--device", default="cuda", help="Device to use for evaluation (cuda or cpu).")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        return
    if not os.path.exists(args.test_file):
        print(f"Error: Test file '{args.test_file}' does not exist.")
        return
    if args.device not in ['cuda', 'cpu']:
        print(f"Error: Invalid device '{args.device}'. Must be 'cuda' or 'cpu'.")
        return

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = 'cpu'

    evaluate_model(args.model_path, args.test_file, args.tokenizer_name, args.device)

if __name__ == "__main__":
    main()
