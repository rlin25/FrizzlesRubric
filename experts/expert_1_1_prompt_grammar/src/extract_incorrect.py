import pandas as pd
import os

SRC = os.path.join(os.path.dirname(__file__), '../data/jfleg_validation.csv')
DST = os.path.join(os.path.dirname(__file__), '../data/jfleg_incorrect.csv')

def main():
    df = pd.read_csv(SRC)
    # Only extract the original sentence as 'prompt', label=0
    out = pd.DataFrame({'prompt': df['sentence'], 'label': 0})
    out.to_csv(DST, index=False)
    print(f"Extracted {len(out)} incorrect prompts to {DST}")

if __name__ == '__main__':
    main() 