import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random
from typing import List, Tuple
import spacy
from googletrans import Translator
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import csv

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

class TextAugmenter:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.translator = Translator()
        self.supported_languages = ['es', 'fr', 'de', 'it']  # Languages for back-translation
        self.augmentation_methods = [
            self.synonym_replacement,
            self.back_translation,
            self.structure_variation,
            self.punctuation_variation
        ]

    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.append(lemma.name())
        return list(set(synonyms))

    def synonym_replacement(self, text: str, replacement_prob: float = 0.3) -> str:
        """Replace words with their synonyms."""
        words = word_tokenize(text)
        augmented_words = []
        
        for word in words:
            if random.random() < replacement_prob:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    augmented_words.append(random.choice(synonyms))
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)

    def back_translation(self, text: str) -> str:
        """Translate text to another language and back to English."""
        try:
            # Translate to a random language
            target_lang = random.choice(self.supported_languages)
            translated = self.translator.translate(text, dest=target_lang)
            
            # Translate back to English
            back_translated = self.translator.translate(translated.text, dest='en')
            return back_translated.text
        except Exception as e:
            print(f"Back translation failed: {e}")
            return text

    def structure_variation(self, text: str) -> str:
        """Vary sentence structure while maintaining grammatical correctness."""
        doc = self.nlp(text)
        
        # Convert active to passive or vice versa
        if random.random() < 0.5:
            # Simple active to passive conversion for basic sentences
            for token in doc:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    # This is a simplified version - in practice, you'd want more complex rules
                    return f"{token.head.text} by {token.text}"
        
        return text

    def punctuation_variation(self, text: str) -> str:
        """Add or remove punctuation marks."""
        # Add random punctuation at the end if none exists
        if not text[-1] in '.!?':
            text += random.choice(['.', '!', '?'])
        
        # Randomly add commas
        words = text.split()
        if len(words) > 5 and random.random() < 0.3:
            insert_pos = random.randint(2, len(words) - 2)
            words.insert(insert_pos, ',')
        
        return ' '.join(words)

    def augment_text(self, text: str, label: int) -> List[Tuple[str, int]]:
        """Apply augmentation techniques to generate new examples."""
        augmented_texts = []
        
        # Only augment negative examples (label 0)
        if label == 0:
            # Randomly select 1-2 augmentation methods
            num_methods = random.randint(1, 2)
            selected_methods = random.sample(self.augmentation_methods, num_methods)
            
            for method in selected_methods:
                augmented_text = method(text)
                if augmented_text != text:  # Only add if the text was actually modified
                    augmented_texts.append((augmented_text, label))
        
        return augmented_texts

def augment_dataset(texts: List[str], labels: List[int], target_ratio: float = 0.6) -> Tuple[List[str], List[int]]:
    """
    Augment the dataset to achieve a target ratio of positive to negative examples.
    
    Args:
        texts: List of input texts
        labels: List of labels (0 for negative, 1 for positive)
        target_ratio: Target ratio of positive examples (default: 0.6)
    """
    augmenter = TextAugmenter()
    augmented_texts = []
    augmented_labels = []
    
    # Calculate current class distribution
    texts = np.array(texts)
    labels = np.array(labels)
    positive_mask = labels == 1
    negative_mask = labels == 0
    
    n_positive = np.sum(positive_mask)
    n_negative = np.sum(negative_mask)
    
    # Calculate how many negative examples we need to achieve target ratio
    target_negative = int(n_positive * (1 - target_ratio) / target_ratio)
    n_negative_to_generate = target_negative - n_negative
    
    if n_negative_to_generate > 0:
        # Randomly select negative examples to augment
        negative_indices = np.where(negative_mask)[0]
        selected_indices = np.random.choice(
            negative_indices,
            size=min(len(negative_indices), n_negative_to_generate),
            replace=False
        )
        
        # Add all original examples
        augmented_texts.extend([str(t) for t in texts])
        augmented_labels.extend(labels)
        
        # Augment selected negative examples with progress bar
        for idx in tqdm(selected_indices, desc="Augmenting negative examples"):
            # Ensure text is a Python str
            new_examples = augmenter.augment_text(str(texts[idx]), int(labels[idx]))
            for new_text, new_label in new_examples:
                augmented_texts.append(str(new_text))
                augmented_labels.append(int(new_label))
    else:
        # If we already have enough negative examples, just return the original dataset
        augmented_texts = [str(t) for t in texts.tolist()]
        augmented_labels = [int(l) for l in labels.tolist()]
    
    return augmented_texts, augmented_labels

def load_data_from_csv(data_dir: str, split: str = "train") -> Tuple[List[str], List[int]]:
    """Load data from a CSV split (train/val/test) in the processed_data directory."""
    data_dir = Path(data_dir)
    csv_file = data_dir / f"{split}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"{csv_file} not found.")
    df = pd.read_csv(csv_file)
    texts = df['sentence'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

def save_augmented_data(texts: List[str], labels: List[int], output_dir: str):
    """Save augmented data to CSV file with columns 'prompt' and 'label'.
    Only the prompt field (below the header) is quoted, header and label are not quoted."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "augmented_data.csv"

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write('prompt,label\n')  # Write header without quotes
        for text, label in zip(texts, labels):
            # Replace internal double quotes with single quotes
            cleaned_text = str(text).replace('"', "'")
            # Surround prompt with double quotes, label as int (never quoted)
            f.write(f'"{cleaned_text}",{int(label)}\n')
    print(f"Augmented data saved to {output_file}")

def print_dataset_stats(texts: List[str], labels: List[int], prefix: str = ""):
    """Print statistics about the dataset."""
    n_total = len(texts)
    n_positive = sum(1 for label in labels if label == 1)
    n_negative = sum(1 for label in labels if label == 0)
    
    print(f"\n{prefix}Dataset Statistics:")
    print(f"Total examples: {n_total}")
    print(f"Positive examples: {n_positive} ({n_positive/n_total*100:.2f}%)")
    print(f"Negative examples: {n_negative} ({n_negative/n_total*100:.2f}%)")

def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'processed_data'
    output_dir = base_dir / 'processed_data' / 'augmented'
    
    # Load data from train.csv
    print("Loading data from train.csv...")
    texts, labels = load_data_from_csv(data_dir, split="train")
    
    # Print original dataset statistics
    print_dataset_stats(texts, labels, "Original")
    
    # Augment dataset
    print("\nAugmenting dataset...")
    augmented_texts, augmented_labels = augment_dataset(texts, labels, target_ratio=0.6)
    
    # Print augmented dataset statistics
    print_dataset_stats(augmented_texts, augmented_labels, "Augmented")
    
    # Save augmented data
    print("\nSaving augmented data...")
    save_augmented_data(augmented_texts, augmented_labels, output_dir)
    print(f"Augmented data saved to {output_dir}")

if __name__ == "__main__":
    main() 