# src/evaluate.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from src.data_preprocessing import preprocess_data

# Load tokenizer and model from trained directory
model_dir = "./models/prompt_clarity_model"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load and prepare the dataset
dataset = preprocess_data("data/prompt_clarity_dataset_clean.csv")
test_dataset = dataset['test']
texts = dataset['test']['text']  # Keep text for viewing predictions

# Setup Trainer
training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
)

# Predict
print("Running evaluation...")
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids

# Evaluate
print("\n📊 Evaluation Metrics:")
print("Accuracy:", accuracy_score(true_labels, pred_labels))
print(classification_report(true_labels, pred_labels, digits=3))

# Show sample predictions
print("\n🔍 Sample Predictions:")
for i in range(10):
    print(f"[{i+1}] Prompt: {texts[i]}")
    print(f"    ➤ Predicted: {pred_labels[i]}, Actual: {true_labels[i]}\n")
