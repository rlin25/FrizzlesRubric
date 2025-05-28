import re
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK stopwords are downloaded
import nltk
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

EN_STOPWORDS = set(stopwords.words('english'))

def preprocess_prompt(prompt: str) -> str:
    """
    Lowercase, remove punctuation, extra whitespace, stopwords, normalize unicode.
    """
    # Normalize unicode
    prompt = unicodedata.normalize('NFKC', prompt)
    # Lowercase
    prompt = prompt.lower()
    # Remove punctuation
    prompt = re.sub(r'[^\w\s]', '', prompt)
    # Remove extra whitespace
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    # Remove stopwords
    tokens = word_tokenize(prompt)
    tokens = [t for t in tokens if t not in EN_STOPWORDS]
    return ' '.join(tokens) 