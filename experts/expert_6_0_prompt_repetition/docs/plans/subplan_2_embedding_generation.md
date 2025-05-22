# Subplan 2: Embedding Generation

## Description
Generate a semantic embedding for the preprocessed prompt using a HuggingFace model.

## Steps
- Load HuggingFace `sentence-transformers/all-MiniLM-L6-v2` model (download if not present)
- Input: preprocessed prompt (string)
- Generate embedding (list of floats) using model.encode()
- Optionally support batching for multiple prompts
- Handle model loading errors and invalid input

## Input
- Preprocessed prompt (string)

## Output
- Embedding (list of floats)

## Pseudocode
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text).tolist()
``` 