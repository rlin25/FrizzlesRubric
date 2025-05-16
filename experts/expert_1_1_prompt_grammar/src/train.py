import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from model import PromptGrammarClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrammarDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer: DistilBertTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def train_model(
    model: PromptGrammarClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01
) -> dict:
    """
    Train the model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        
    Returns:
        dict: Training history
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                val_steps += 1
                
                preds = (outputs.squeeze() > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / val_steps
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save("models/best_model")
            logger.info("Saved new best model")
    
    return history

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    data_dir = Path("processed_data")
    train_data = pd.read_csv(data_dir / "train.csv")
    val_data = pd.read_csv(data_dir / "val.csv")
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = PromptGrammarClassifier()
    
    # Create datasets
    train_dataset = GrammarDataset(
        train_data['sentence'].values,
        train_data['label'].values,
        tokenizer
    )
    val_dataset = GrammarDataset(
        val_data['sentence'].values,
        val_data['label'].values,
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train model
    history = train_model(model, train_loader, val_loader, device)
    
    # Save training history
    with open("models/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 