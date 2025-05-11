# src/train.py

from transformers import Trainer, TrainingArguments
from model import create_model
from data_preprocessing import preprocess_data

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

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained('./models/prompt_clarity_model')

if __name__ == "__main__":
    # Load and preprocess data
    dataset = preprocess_data("data/prompt_clarity_dataset_clean.csv")
    
    # Train the model
    train_model(dataset)
