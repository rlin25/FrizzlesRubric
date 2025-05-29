import os
import argparse
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob

class PromptDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])  # Ensure label is an integer
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PromptClassifierTrainer:
    def __init__(self, model_name: str = "bert-base-uncased", output_dir: str = "models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        ).to(self.device)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped output directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # predictions and labels are both 1D arrays of class indices
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions)
        }
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        n_epochs: int = 5,
        batch_size: int = 8,  # Reduced batch size
        gradient_accumulation_steps: int = 2,  # Added gradient accumulation
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        early_stopping_patience: int = 3
    ) -> Dict[str, List[float]]:
        # Create datasets
        train_dataset = PromptDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = PromptDataset(val_texts, val_labels, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(train_loader) * n_epochs // gradient_accumulation_steps
        )
        
        # Training loop
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            optimizer.zero_grad()  # Reset gradients at the start of epoch
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")
            for i, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps  # Normalize loss
                
                loss.backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
            
            avg_train_loss = train_loss / len(train_loader)
            metrics['train_loss'].append(avg_train_loss)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            metrics['val_loss'].append(val_metrics['loss'])
            metrics['val_f1'].append(val_metrics['f1'])
            
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                self.save('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = self.compute_metrics((np.array(all_preds), np.array(all_labels)))
        metrics['loss'] = val_loss / len(val_loader)
        
        return metrics
    
    def save(self, filename: str):
        """Save model and tokenizer"""
        save_path = self.run_dir / filename
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

def load_data(data_dir):
    all_texts = []
    all_labels = []
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    for csv_file in csv_files:
        try:
            # Read CSV with proper quoting and error handling
            df = pd.read_csv(csv_file, quoting=1, on_bad_lines='skip')  # QUOTE_ALL mode
            # Ensure labels are integers
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
            all_texts.extend(df['prompt'].tolist())
            all_labels.extend(df['label'].tolist())
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
            continue
    
    return all_texts, all_labels

def main():
    parser = argparse.ArgumentParser(description='Train prompt classifier')
    parser.add_argument('--data_dir', type=str, default='data/prompts/original_gemini',
                      help='Directory containing the prompt CSV files')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save the trained model')
    parser.add_argument('--n_epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Training batch size')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                      help='Number of epochs to wait for improvement')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    texts, labels = load_data(args.data_dir)
    
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    
    # Initialize trainer
    trainer = PromptClassifierTrainer(output_dir=args.output_dir)
    
    # Train model
    print("Training model...")
    history = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main() 