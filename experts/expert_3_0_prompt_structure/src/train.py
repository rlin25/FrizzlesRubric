import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
from model import DistilBertFileClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

class FileRefDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

def train_model(data_path, model_save_path, epochs=3, batch_size=16, lr=2e-5, device='cpu'):
    df = pd.read_csv(data_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['prompt'], df['label'], test_size=0.2, stratify=df['label'])
    model = DistilBertFileClassifier().to(device)
    tokenizer = model.tokenizer
    train_ds = FileRefDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_ds = FileRefDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f'Training Epoch {epoch+1}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask).squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dl)
        print(f'Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                logits = model(input_ids, attention_mask).squeeze(-1)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dl)
        print(f'Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pt'))
            print('Best model saved.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to CSV file with text and label columns')
    parser.add_argument('--model_save_path', required=True, help='Directory to save the best model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train_model(args.data_path, args.model_save_path, args.epochs, args.batch_size, args.lr, args.device) 