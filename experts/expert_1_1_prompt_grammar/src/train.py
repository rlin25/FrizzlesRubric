import os
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from model import create_model
from data_preprocessing import preprocess_data
import logging
from transformers import DistilBertForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(dataset, model_name="distilbert-base-uncased"):
    """Train the grammar classifier model."""
    # Split the dataset
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]  # Use validation set for evaluation
    test_dataset = dataset["test"]  # Keep test set for final evaluation
    
    # Calculate class weights
    labels = np.asarray(train_dataset["label"])
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
        
    # Create model (HuggingFace compatible)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/grammar_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        logging_dir="./logs",
        save_total_limit=2,
        remove_unused_columns=True,
        fp16=False,
        save_strategy="steps",
        save_steps=100
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")
    
    # Save the final model
    trainer.save_model("./models/grammar_model")
    logger.info("Training completed and model saved.")

if __name__ == "__main__":
    # Get the absolute path to the processed_data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../processed_data/")
    
    # Load and preprocess the dataset
    dataset = preprocess_data(data_dir)
    
    # Train the model
    train_model(dataset) 