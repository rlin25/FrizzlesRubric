import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_on_live_example(model_dir, example_text, device='cpu'):
    """Test the model on a single example using HuggingFace Transformers."""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        prediction = prediction.item()
        confidence = confidence.item()

    return prediction, confidence

if __name__ == "__main__":
    # Model settings
    model_dir = "/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/models/grammar_model"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Enter a prompt to test the grammar model (type 'exit' to quit):")
    while True:
        text = input("Prompt: ")
        if text.strip().lower() == 'exit':
            break
        prediction, confidence = test_model_on_live_example(model_dir, text, device=device)
        print(f"\nInput Text: {text}")
        print(f"Predicted Class: {prediction}")
        print(f"Confidence: {confidence:.2f}") 