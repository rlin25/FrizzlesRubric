import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from typing import List, Dict, Tuple
import pandas as pd

class GranularityDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = 512):
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

def load_data(file_path: str) -> Tuple[List[str], List[int]]:
    """Load data from CSV file."""
    df = pd.read_csv(file_path)
    return df['prompt'].values.tolist(), df['label'].values.tolist()

def create_data_loaders(
    texts: List[str],
    labels: List[int],
    tokenizer: DistilBertTokenizer,
    batch_size: int = 32,
    train_ratio: float = 0.8
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders."""
    # Split data
    train_size = int(len(texts) * train_ratio)
    train_texts, val_texts = texts[:train_size], texts[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]

    # Create datasets
    train_dataset = GranularityDataset(train_texts, train_labels, tokenizer)
    val_dataset = GranularityDataset(val_texts, val_labels, tokenizer)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader 