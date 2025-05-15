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

    def prepare_data(self, train_path, val_path, batch_size=16):
        """Prepare data loaders for training and validation."""
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

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
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        return train_loader, val_loader

    def train(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
        """Train the model."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_f1 = 0
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")

            # Validation
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Validation metrics: {val_metrics}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_model()

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

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average='binary'
        )
        accuracy = accuracy_score(true_labels, predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_model(self):
        """Save the model and tokenizer."""
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")

if __name__ == "__main__":
    trainer = GrammarTrainer()
    train_loader, val_loader = trainer.prepare_data(
        "processed_data/train.csv",
        "processed_data/val.csv"
    )
    trainer.train(train_loader, val_loader) 