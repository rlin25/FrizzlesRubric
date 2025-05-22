import glob
import os
from transformers import Trainer, TrainingArguments
from model import create_model
from data_preprocessing import preprocess_data
import pandas as pd
from transformers import AutoModelForSequenceClassification

def merge_datasets(dataset_paths, data_dir=None):
    """Function to merge multiple datasets into one."""
    # Load and merge all datasets
    datasets = [pd.read_csv(path) for path in dataset_paths]
    merged_dataset = pd.concat(datasets, ignore_index=True)
    
    # Save merged dataset to CSV
    if data_dir is not None:
        merged_path = os.path.join(data_dir, "clarity_merged.csv")
    else:
        merged_path = "clarity_merged.csv"
    merged_dataset.to_csv(merged_path, index=False)  # Save to CSV
    
    return merged_dataset

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint by looking at the folder names."""
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        return None  # No checkpoints found
    
    # Extract the numeric part of each checkpoint folder (e.g., '28' from 'checkpoint-28')
    checkpoint_nums = [int(d.split('-')[1]) for d in checkpoint_dirs]
    
    # Find the latest checkpoint (the one with the highest number)
    latest_checkpoint = max(checkpoint_nums)
    
    return os.path.join(output_dir, f'checkpoint-{latest_checkpoint}')

def train_model(dataset, model_name="distilbert-base-uncased"):
    # Split the dataset into training and testing sets
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Set up the model with explicit configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        cache_dir="./cache"  # Use a separate cache directory
    )

    # Define training arguments with more conservative settings
    training_args = TrainingArguments(
        output_dir="./models/prompt_clarity_model",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",        # Save at the end of each epoch
        learning_rate=1e-5,           # Slightly lower learning rate
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=8,   # Smaller batch size
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,             # Log less frequently
        logging_dir="./logs",
        save_total_limit=2,           # Keep only the last 2 checkpoints
        remove_unused_columns=True,   # Remove unused columns to save memory
        fp16=False                    # Disable mixed precision training
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Force fresh start by removing any existing checkpoints and cache
    if os.path.exists('./models/prompt_clarity_model'):
        import shutil
        shutil.rmtree('./models/prompt_clarity_model')
    if os.path.exists('./cache'):
        shutil.rmtree('./cache')
    os.makedirs('./models/prompt_clarity_model', exist_ok=True)
    os.makedirs('./cache', exist_ok=True)

    print("Starting fresh training with clean initialization...")
    trainer.train()

    # Save the final model
    trainer.save_model('./models/prompt_clarity_model')

if __name__ == "__main__":
    # Get the absolute path to the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data/")
    dataset_paths = [
        path for path in glob.glob(f"{data_dir}**/clarity_*.csv", recursive=True)
        if not path.endswith("clarity_merged.csv")
    ]

    # Merge the datasets
    merged_dataset = merge_datasets(dataset_paths, data_dir=data_dir)
    
    # Preprocess merged dataset
    dataset = preprocess_data(merged_dataset)
    
    # Train the model
    train_model(dataset)