import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/jfleg.csv')
SEED = 42

# Load JFLEG dataset (assume CSV with columns: 'prompt', 'label')
def load_jfleg_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def balance_dataset(df):
    # Ensure equal number of positive and negative examples
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    n = min(len(pos), len(neg))
    pos_sample = pos.sample(n, random_state=SEED)
    neg_sample = neg.sample(n, random_state=SEED)
    balanced = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return balanced

def split_dataset(df):
    # Stratified split: 80% train, 10% val, 10% test
    train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=SEED)
    return train, val, test

if __name__ == '__main__':
    df = load_jfleg_dataset()
    balanced = balance_dataset(df)
    train, val, test = split_dataset(balanced)
    train.to_csv('../data/train.csv', index=False)
    val.to_csv('../data/val.csv', index=False)
    test.to_csv('../data/test.csv', index=False)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}") 