import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_grammar_model(model_path, text):
    """Test the grammar model on a single example."""
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {model_path}")
    
    # Load the model from safetensors
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0].item()
        prediction = torch.argmax(probs, dim=1).item()
    
    return prediction, confidence

if __name__ == "__main__":
    # Model path
    model_path = "/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/models/grammar_model/checkpoint-987"
    logger.info(f"Using model path: {model_path}")
    
    # Example texts to test
    test_texts = [
        "Fix this code.",  # Poor grammar
        "Create a RESTful API in Flask that accepts user input and stores it in a PostgreSQL database.",  # Good grammar
        "Correct this stupid thing. I've been trying to fix it for hours. I don't know what to do.",  # Poor grammar
        "Scrape the NBA website into a CSV file via BeautifulSoup.",  # Good grammar
        "Fix code this.",  # Poor grammar
        "Create a API RESTful in Flask that user input accepts and itstores in a databasePostgreSQL.",  # Poor grammar
        "Correct this stupid thing. trying trying I've been to for fix it hours. I whatdon't know to do.",  # Poor grammar
        "Scrape the website intoNBA a BeautifulSoupfile CSV via."  # Poor grammar
    ]
    
    # Test each example
    for text in test_texts:
        prediction, confidence = test_grammar_model(model_path, text)
        print(f"\nInput Text: {text}")
        print(f"Predicted Class: {'Correct' if prediction == 1 else 'Incorrect'}")
        print(f"Confidence: {confidence:.2f}") 