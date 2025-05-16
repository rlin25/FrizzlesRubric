from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from pathlib import Path

def download_model():
    print("Downloading DistilBERT model and tokenizer...")
    
    # Initialize model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary classification for grammar correctness
    )
    
    # Create models directory if it doesn't exist
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    download_model() 