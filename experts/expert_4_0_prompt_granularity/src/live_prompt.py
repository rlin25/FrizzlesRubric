import torch
from pathlib import Path
from model import GranularityClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initialize model
    model = GranularityClassifier()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load the best model checkpoint
    checkpoint_path = Path('/home/ubuntu/FrizzlesRubric/experts/expert_4_0_prompt_granularity/models/checkpoints/best_model.pt')
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        exit(1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Loaded model from {checkpoint_path}")

    # Live prompt mode
    print("\nEnter your own prompts for live testing (type 'exit' to quit):")
    while True:
        user_input = input("Prompt: ")
        if user_input.strip().lower() == "exit":
            break
        pred, conf = model.predict(user_input, device=device)
        print(f"Prediction: {pred:.4f} (Confidence: {conf:.2f})") 