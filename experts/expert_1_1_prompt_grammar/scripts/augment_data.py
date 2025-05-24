import pandas as pd
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_augmentation import augment_dataset

def main():
    # Load the training data
    data_dir = Path(__file__).parent.parent / 'processed_data'
    train_df = pd.read_csv(data_dir / 'train.csv')
    
    print("Original dataset size:", len(train_df))
    print("Class distribution before augmentation:")
    print(train_df['label'].value_counts())
    
    # Apply augmentation with progress bar
    print("\nStarting augmentation process...")
    augmented_texts = []
    augmented_labels = []
    
    # Process in batches to show progress
    batch_size = 100
    for i in tqdm(range(0, len(train_df), batch_size), desc="Augmenting data"):
        batch_texts = train_df['sentence'].iloc[i:i+batch_size].tolist()
        batch_labels = train_df['label'].iloc[i:i+batch_size].tolist()
        
        batch_augmented_texts, batch_augmented_labels = augment_dataset(batch_texts, batch_labels)
        augmented_texts.extend(batch_augmented_texts)
        augmented_labels.extend(batch_augmented_labels)
        
        # Print intermediate progress
        if (i + batch_size) % 500 == 0:
            print(f"\nProcessed {min(i + batch_size, len(train_df))}/{len(train_df)} examples")
            print(f"Current augmented dataset size: {len(augmented_texts)}")
    
    # Create new DataFrame with augmented data
    augmented_df = pd.DataFrame({
        'sentence': augmented_texts,
        'label': augmented_labels
    })
    
    print("\nAugmentation complete!")
    print("Final dataset size:", len(augmented_df))
    print("Class distribution after augmentation:")
    print(augmented_df['label'].value_counts())
    
    # Save augmented dataset
    output_path = data_dir / 'train_augmented.csv'
    augmented_df.to_csv(output_path, index=False)
    print(f"\nAugmented dataset saved to: {output_path}")

if __name__ == "__main__":
    main() 