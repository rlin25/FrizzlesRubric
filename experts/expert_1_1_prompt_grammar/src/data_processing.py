import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JFLEGProcessor:
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def load_jfleg(self) -> None:
        """Load the JFLEG dataset."""
        logger.info("Loading JFLEG dataset...")
        dataset = load_dataset("jfleg")
        
        # Debug: Print dataset structure
        logger.info("Dataset structure:")
        logger.info(f"Keys: {dataset.keys()}")
        logger.info(f"Validation set columns: {dataset['validation'].column_names}")
        logger.info(f"Test set columns: {dataset['test'].column_names}")
        
        # Convert to pandas DataFrame for easier processing
        train_data = pd.DataFrame(dataset['validation'])
        test_data = pd.DataFrame(dataset['test'])
        
        # Debug: Print DataFrame info
        logger.info("\nTrain data info:")
        logger.info(train_data.info())
        logger.info("\nTest data info:")
        logger.info(test_data.info())
        
        # Combine train and test data
        self.data = pd.concat([train_data, test_data], ignore_index=True)
        logger.info(f"Loaded {len(self.data)} samples")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare the data for training, validation, and testing."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_jfleg first.")

        # Create binary classification dataset
        # Original sentences are class 0 (incorrect)
        incorrect_samples = self.data[['sentence']].copy()
        incorrect_samples['label'] = 0

        # Reference sentences are class 1 (correct)
        correct_samples = []
        for _, row in self.data.iterrows():
            for ref in row['corrections']:
                correct_samples.append({
                    'sentence': ref,
                    'label': 1
                })
        correct_samples = pd.DataFrame(correct_samples)

        # Combine and shuffle
        combined_data = pd.concat([incorrect_samples, correct_samples], ignore_index=True)
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train, validation, and test sets
        train_data, temp_data = train_test_split(combined_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        logger.info(f"Train set size: {len(train_data)}")
        logger.info(f"Validation set size: {len(val_data)}")
        logger.info(f"Test set size: {len(test_data)}")

        return train_data, val_data, test_data

    def save_splits(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Save the processed data splits."""
        train_data.to_csv(self.output_dir / "train.csv", index=False)
        val_data.to_csv(self.output_dir / "val.csv", index=False)
        test_data.to_csv(self.output_dir / "test.csv", index=False)
        
        # Save dataset statistics
        stats = {
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "train_class_distribution": train_data['label'].value_counts().to_dict(),
            "val_class_distribution": val_data['label'].value_counts().to_dict(),
            "test_class_distribution": test_data['label'].value_counts().to_dict()
        }
        
        with open(self.output_dir / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    processor = JFLEGProcessor()
    processor.load_jfleg()
    train_data, val_data, test_data = processor.prepare_data()
    processor.save_splits(train_data, val_data, test_data) 