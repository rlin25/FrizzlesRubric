import logging
from src.keywords import get_keywords

def detect_ai_tooling(prompt: str, keywords=None) -> int:
    if keywords is None:
        keywords = get_keywords()
    normalized_prompt = prompt.lower().strip()
    for keyword in keywords:
        if keyword in normalized_prompt:
            logging.info(f"Detected keyword '{keyword}' in prompt.")
            return 0
    logging.info("No AI tooling detected in prompt.")
    return 1 