import glob
import os
from transformers import Trainer, TrainingArguments
from model import create_model
from data_preprocessing import preprocess_data
import pandas as pd

def merge_datasets(dataset_paths):
    """Function to merge multiple datasets into one."""
    # Load and merge all datasets
    datasets = [pd.read_csv(path) for path in dataset_paths]
    merged_dataset = pd.concat(datasets, ignore_index=True)
    
    # Save merged dataset to CSV
    merged_dataset.to_csv("data/clarity_merged.csv", index=False)  # Save to CSV
    
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

def train_model(dataset, model_name="bert-base-uncased"):
    # Split the dataset into training and testing sets
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Set up the model
    model = create_model(model_name=model_name)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models/prompt_clarity_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",  # ✅ Match eval strategy
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Get the latest checkpoint
    checkpoint = get_latest_checkpoint('./models/prompt_clarity_model')

    # Train the model (resume if a checkpoint exists)
    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save the model
    model.save_pretrained('./models/prompt_clarity_model')

if __name__ == "__main__":
    # Automatically collect all CSV files in the `data/` directory that match clarity datasets
    data_dir = "data/"
    dataset_paths = [
        path for path in glob.glob("data/**/clarity_*.csv", recursive=True)
        if not path.endswith("clarity_merged.csv")
    ]



    # Merge the datasets
    merged_dataset = merge_datasets(dataset_paths)
    
    # Preprocess merged dataset
    dataset = preprocess_data(merged_dataset)
    
    # Train the model
    train_model(dataset)