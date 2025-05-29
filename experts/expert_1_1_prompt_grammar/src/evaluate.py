import os
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model_checkpoint')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/test.csv')
MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict_batch(prompts):
    inputs = tokenizer(prompts, return_tensors='pt', truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
    return preds

def main():
    df = pd.read_csv(DATA_PATH)
    y_true = df['label'].values
    y_pred = []
    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch_prompts = df['prompt'].iloc[i:i+batch_size].tolist()
        batch_preds = predict_batch(batch_prompts)
        y_pred.extend(batch_preds)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)

if __name__ == '__main__':
    main() 