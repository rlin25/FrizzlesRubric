from sentence_transformers import SentenceTransformer
from typing import List

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def generate_embedding(prompt: str) -> List[float]:
    model = get_model()
    embedding = model.encode(prompt)
    return embedding.tolist() 