import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_NAME = 'distilbert-base-uncased'
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '../model_checkpoint')
SEED = 42

# Load positive and negative examples
pos_df = pd.read_csv(os.path.join(DATA_DIR, 'jfleg_first_positive_training.csv'))
neg_df = pd.read_csv(os.path.join(DATA_DIR, 'jfleg_incorrect.csv'))

# Combine and shuffle
all_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Stratified split: 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(all_df, test_size=0.2, stratify=all_df['label'], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)

def df_to_hf_dataset(df):
    return Dataset.from_pandas(df[['prompt', 'label']])

train_dataset = df_to_hf_dataset(train_df)
val_dataset = df_to_hf_dataset(val_df)
test_dataset = df_to_hf_dataset(test_df)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    prompts = batch['prompt']
    if isinstance(prompts, str):
        prompts = [prompts]
    # Filter out or replace non-string values
    clean_prompts = [p if isinstance(p, str) and p.strip() != '' else '' for p in prompts]
    for i, p in enumerate(clean_prompts):
        if not isinstance(p, str) or p == '':
            print(f'REMOVED/EMPTY at index {i}: {repr(p)} (type: {type(p)})')
    print('DEBUG: prompts type:', type(clean_prompts), 'first 5:', clean_prompts[:5])
    return tokenizer(clean_prompts, padding='max_length', truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    seed=SEED,
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    cm = confusion_matrix(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

if __name__ == '__main__':
    print('Starting training...')
    trainer.train()
    print('Training complete. Saving model...')
    trainer.save_model(CHECKPOINT_DIR)
    print('Model saved to', CHECKPOINT_DIR) 