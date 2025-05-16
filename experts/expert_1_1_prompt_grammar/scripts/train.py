import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from sklearn.model_selection import KFold
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrammarDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

class GrammarTrainer:
    def __init__(self, model_name="distilbert-base-uncased", output_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, train_path, val_path, batch_size=32):
        """Prepare data loaders for training and validation."""
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # Calculate class weights
        class_counts = train_df['label'].value_counts()
        total_samples = len(train_df)
        class_weights = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }
        self.class_weights = torch.tensor([class_weights[0], class_weights[1]]).to(self.device)

        train_dataset = GrammarDataset(
            train_df['sentence'].values,
            train_df['label'].values,
            self.tokenizer
        )
        val_dataset = GrammarDataset(
            val_df['sentence'].values,
            val_df['label'].values,
            self.tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4
        )

        return train_loader, val_loader

    def train(self, train_loader, val_loader, num_epochs=5, learning_rate=2e-5, patience=3):
        """Train the model with early stopping and improved scheduling."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs
        
        # Warmup for 10% of total steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_f1 = 0
        patience_counter = 0
        training_stats = []

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Apply class weights to loss
                loss = outputs.loss * self.class_weights[labels].mean()
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Collect predictions
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            train_metrics = self.calculate_metrics(train_labels, train_preds)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            training_stats.append(epoch_stats)
            
            logger.info(f"Epoch {epoch + 1} stats:")
            logger.info(f"Train loss: {avg_train_loss:.4f}")
            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Validation metrics: {val_metrics}")

            # Save best model and check early stopping
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_model()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Save training statistics
        with open(self.run_dir / 'training_stats.json', 'w') as f:
            json.dump(training_stats, f, indent=2)

    def calculate_metrics(self, true_labels, predictions):
        """Calculate classification metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average='binary'
        )
        accuracy = accuracy_score(true_labels, predictions)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.numpy())

        return self.calculate_metrics(true_labels, predictions)

    def save_model(self):
        """Save the model and tokenizer."""
        model_path = self.run_dir / "best_model"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"Model saved to {model_path}")

def cross_validate(train_df, n_splits=5):
    """Perform k-fold cross validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        logger.info(f"Training fold {fold + 1}/{n_splits}")
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Save fold data
        fold_dir = Path("processed_data") / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        train_fold.to_csv(fold_dir / "train.csv", index=False)
        val_fold.to_csv(fold_dir / "val.csv", index=False)
        
        # Train model
        trainer = GrammarTrainer(output_dir=f"models/fold_{fold}")
        train_loader, val_loader = trainer.prepare_data(
            fold_dir / "train.csv",
            fold_dir / "val.csv"
        )
        trainer.train(train_loader, val_loader)
        
        # Evaluate on validation set
        metrics = trainer.evaluate(val_loader)
        fold_metrics.append(metrics)
        
        logger.info(f"Fold {fold + 1} metrics: {metrics}")

    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    logger.info(f"Average metrics across folds: {avg_metrics}")
    
    return fold_metrics, avg_metrics

if __name__ == "__main__":
    # Load full training data
    train_df = pd.read_csv("processed_data/train.csv")
    
    # Perform cross-validation
    fold_metrics, avg_metrics = cross_validate(train_df)
    
    # Train final model on full dataset
    logger.info("Training final model on full dataset")
    trainer = GrammarTrainer(output_dir="models/final")
    train_loader, val_loader = trainer.prepare_data(
        "processed_data/train.csv",
        "processed_data/val.csv"
    )
    trainer.train(train_loader, val_loader) 