import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import logging
from pathlib import Path
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_model():
    model_dir = Path("models/grammar_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading base model and tokenizer...")
    # Load the base model and tokenizer
    base_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "incorrect", 1: "correct"},
        label2id={"incorrect": 0, "correct": 1}
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    logger.info("Loading trained weights...")
    # Load the trained weights using safetensors
    state_dict = load_file(model_dir / "model.safetensors", device="cpu")
    
    # Update the model with trained weights
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    base_model.eval()
    
    logger.info("Saving fixed model...")
    # Save the fixed model
    base_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    logger.info("Model fixed successfully!")

if __name__ == "__main__":
    fix_model() 