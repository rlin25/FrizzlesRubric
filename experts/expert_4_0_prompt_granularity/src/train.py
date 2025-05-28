import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd
import os
from datetime import datetime

from model import GranularityClassifier
from dataset import GranularityDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create checkpoint directory
CHECKPOINT_DIR = Path('models/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(data_path: str):
    """Load data from CSV file."""
    df = pd.read_csv(data_path)
    return df['prompt'].tolist(), df['label'].tolist()

def create_data_loaders(texts, labels, tokenizer, batch_size=16, val_split=0.2):
    """Create train and validation data loaders."""
    # Split data into train and validation sets
    val_size = int(len(texts) * val_split)
    train_texts, val_texts = texts[val_size:], texts[:val_size]
    train_labels, val_labels = labels[val_size:], labels[:val_size]
    
    # Create datasets
    train_dataset = GranularityDataset(train_texts, train_labels, tokenizer)
    val_dataset = GranularityDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

class Trainer:
    def __init__(
        self,
        model: GranularityClassifier,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        num_epochs: int = 5,
        early_stopping_patience: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.current_epoch = 0
        self.best_f1 = 0
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.criterion = nn.BCELoss()
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            try:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Log shapes for debugging
                if batch_idx == 0:
                    logger.info(f"Batch shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
                
                self.optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                # Reshape labels to match output shape
                labels = labels.unsqueeze(1)  # Add dimension to match [batch_size, 1]
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(f"Batch contents: {batch}")
                raise
            
        return total_loss / len(self.train_loader)
    
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                # Reshape labels to match output shape
                labels = labels.unsqueeze(1)  # Add dimension to match [batch_size, 1]
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions),
            'recall': recall_score(all_labels, all_predictions),
            'f1': f1_score(all_labels, all_predictions)
        }
        
        return metrics['f1'], metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model with early stopping."""
        best_f1 = 0
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            logger.info(f"Epoch {self.current_epoch}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            
            # Evaluate
            val_f1, val_metrics = self.evaluate()
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_f1)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Validation F1: {val_f1:.4f}")
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.best_f1 = best_f1
                patience_counter = 0
                self.save_model('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
        
        return history
    
    def save_model(self, path: str):
        """Save the model and training state."""
        checkpoint_path = CHECKPOINT_DIR / path
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_f1': self.best_f1,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_model(self, path: str):
        """Load the model and training state."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path)
        
        # Handle both old and new checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_f1 = checkpoint['best_f1']
        else:
            # Old format (just model state dict)
            self.model.load_state_dict(checkpoint)
            logger.info("Loaded model from old format checkpoint")
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

def main():
    # Load data
    texts, labels = load_data('experts/expert_4_0_prompt_granularity/data/full_dataset.csv')
    
    # Initialize model and tokenizer
    model = GranularityClassifier()
    tokenizer = model.tokenizer
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(texts, labels, tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        early_stopping_patience=3
    )
    
    # Try to load existing checkpoint
    checkpoint_path = '/home/ubuntu/FrizzlesRubric/experts/expert_4_0_prompt_granularity/best_model.pt'
    if os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint at {checkpoint_path}")
        trainer.load_model(checkpoint_path)
        logger.info(f"Resuming training from epoch {trainer.current_epoch}")
    
    # Train the model
    history = trainer.train()
    
    # Save training history
    history_path = CHECKPOINT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main() 