import pandas as pd
import numpy as np
from pathlib import Path
from src.data_augmentation import TextAugmenter, augment_dataset
import json
from tqdm import tqdm

def balance_dataset(input_file: str, output_file: str, target_ratio: float = 0.5, seed: int = 42):
    """
    Balance the dataset by combining undersampling and augmentation.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output balanced CSV file
        target_ratio: Target ratio of positive examples (default: 0.5)
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Split into correct and incorrect examples
    correct_examples = df[df["label"] == 1]
    incorrect_examples = df[df["label"] == 0]
    
    # Calculate target number of examples for each class
    n_correct = len(correct_examples)
    n_incorrect = len(incorrect_examples)
    
    if n_correct > n_incorrect:
        # Undersample correct examples
        correct_examples = correct_examples.sample(n=n_incorrect, random_state=seed)
        balanced_df = pd.concat([correct_examples, incorrect_examples])
    else:
        # Augment incorrect examples
        texts = incorrect_examples["sentence"].tolist()
        labels = incorrect_examples["label"].tolist()
        
        # Augment to match correct examples count
        augmented_texts, augmented_labels = augment_dataset(
            texts=texts,
            labels=labels,
            target_ratio=target_ratio
        )
        
        # Create new dataframe with augmented examples
        augmented_df = pd.DataFrame({
            "sentence": augmented_texts,
            "label": augmented_labels
        })
        
        # Combine original correct examples with augmented incorrect examples
        balanced_df = pd.concat([correct_examples, augmented_df])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save the balanced dataset
    balanced_df.to_csv(output_file, index=False)
    
    # Calculate and print statistics
    total = len(balanced_df)
    correct = len(balanced_df[balanced_df["label"] == 1])
    incorrect = len(balanced_df[balanced_df["label"] == 0])
    
    stats = {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "correct_ratio": correct/total,
        "incorrect_ratio": incorrect/total
    }
    
    print(f"Balanced dataset statistics for {Path(input_file).name}:")
    print(f"Total examples: {total}")
    print(f"Correct examples: {correct} ({correct/total:.2%})")
    print(f"Incorrect examples: {incorrect} ({incorrect/total:.2%})")
    
    return stats

def main():
    """Process all dataset splits and save statistics."""
    # Create output directory
    output_dir = Path("processed_data/balanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = ["train", "val", "test"]
    all_stats = {}
    
    for split in splits:
        input_file = f"processed_data/{split}.csv"
        output_file = output_dir / f"{split}.csv"
        
        print(f"Processing {split} split...")
        stats = balance_dataset(
            input_file=input_file,
            output_file=str(output_file),
            target_ratio=0.5
        )
        all_stats[split] = stats
    
    # Save statistics
    with open(output_dir / "balancing_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print("Balancing complete! Statistics saved to balancing_stats.json")

if __name__ == "__main__":
    main()
