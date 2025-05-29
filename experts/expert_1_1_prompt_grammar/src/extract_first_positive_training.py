import pandas as pd
import os

SRC = os.path.join(os.path.dirname(__file__), '../data/jfleg_validation.csv')
DST = os.path.join(os.path.dirname(__file__), '../data/jfleg_first_positive_training.csv')

def extract_first_correction(corrections):
    # Split by newline, filter out empty lines, and take the first
    lines = [line.strip() for line in str(corrections).split('\n') if line.strip()]
    if not lines:
        return ''
    first = lines[0]
    # Remove leading/trailing square brackets
    if first.startswith("["):
        first = first[1:]
    if first.endswith("]"):
        first = first[:-1]
    # Remove all double quotes
    first = first.replace('"', '')
    return first.strip()

def main():
    df = pd.read_csv(SRC)
    df['prompt'] = df['corrections'].apply(extract_first_correction)
    df['label'] = 1
    out = df[['prompt', 'label']]
    out = out[out['prompt'] != '']
    out.to_csv(DST, index=False)
    print(f"Extracted {len(out)} first positive prompts to {DST}")

if __name__ == '__main__':
    main() 