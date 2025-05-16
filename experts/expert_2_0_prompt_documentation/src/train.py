import os
import argparse
from pathlib import Path
import pandas as pd
from typing import Tuple

from data.load_preprocess import prepare_dataset
from data.augmentation import PromptAugmenter
from models.binary_classifier import DocumentationClassifierTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train documentation classifier')
    parser.add_argument('--data_dir', type=str, default='data/prompts',
                      help='Directory containing the prompt CSV files')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save the trained model')
    parser.add_argument('--n_epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--n_augmentations', type=int, default=2,
                      help='Number of augmentations per training example')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                      help='Number of epochs to wait for improvement')
    return parser.parse_args()

def prepare_training_data(data_dir: str, n_augmentations: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training data by loading and augmenting the dataset.
    
    Args:
        data_dir (str): Path to the data directory
        n_augmentations (int): Number of augmentations per example
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames
    """
    # Load and split the dataset
    train_df, test_df = prepare_dataset(data_dir)
    
    # Augment training data
    augmenter = PromptAugmenter()
    augmented_texts = augmenter.augment_dataset(
        train_df['text'].tolist(),
        n_augmentations=n_augmentations
    )
    
    # Create augmented DataFrame
    augmented_df = pd.DataFrame({
        'text': augmented_texts,
        'label': [1] * len(augmented_texts)  # All augmented texts are well-documented
    })
    
    # Combine original and augmented data
    train_df = pd.concat([train_df, augmented_df], ignore_index=True)
    
    return train_df, test_df

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing training data...")
    train_df, test_df = prepare_training_data(args.data_dir, args.n_augmentations)
    
    # Initialize trainer
    trainer = DocumentationClassifierTrainer()
    
    # Train model
    print("Training model...")
    history = trainer.train(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label'].tolist(),
        test_texts=test_df['text'].tolist(),
        test_labels=test_df['label'].tolist(),
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, 'documentation_classifier.pt')
    trainer.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

if __name__ == '__main__':
    main() 