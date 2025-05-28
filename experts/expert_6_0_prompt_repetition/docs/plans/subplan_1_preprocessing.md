# Subplan 1: Preprocessing

## Description
Preprocess the raw prompt to normalize and clean the text for embedding generation and similarity checking.

## Steps
- Lowercase the prompt using Python's `str.lower()`
- Remove punctuation using regex (e.g., `re.sub(r'[^\w\s]', '', text)`) or `string.punctuation`
- Remove extra whitespace (e.g., `re.sub(r'\s+', ' ', text).strip()`)
- Remove NLTK English stopwords (download and use `nltk.corpus.stopwords.words('english')`)
- Normalize unicode (e.g., `unicodedata.normalize('NFKC', text)`)
- Handle edge cases: empty prompts, prompts with only stopwords, or only punctuation
- Return error if prompt is empty after preprocessing

## Input
- Raw prompt (string)

## Output
- Preprocessed prompt (string)

## Pseudocode
```python
import re, string, unicodedata
from nltk.corpus import stopwords

def preprocess_prompt(prompt):
    text = prompt.lower()
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([w for w in text.split() if w not in stop_words])
    if not text:
        raise ValueError('Prompt is empty after preprocessing')
    return text
``` 