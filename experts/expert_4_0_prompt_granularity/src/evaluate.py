import torch
from pathlib import Path
from train import Trainer, load_data, create_data_loaders
from model import GranularityClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load data (same split as train.py)
    texts, labels = load_data('experts/expert_4_0_prompt_granularity/data/full_dataset.csv')
    model = GranularityClassifier()
    tokenizer = model.tokenizer
    train_loader, val_loader = create_data_loaders(texts, labels, tokenizer)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,  # Not used, just for init
        early_stopping_patience=1  # Not used, just for init
    )

    # Load the best model checkpoint
    checkpoint_path = Path('experts/expert_4_0_prompt_granularity/models/checkpoints/best_model.pt')
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        exit(1)
    trainer.load_model(str(checkpoint_path))

    # Live prompt mode only
    print("\nEnter your own prompts for live testing (type 'exit' to quit):")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    while True:
        user_input = input("Prompt: ")
        if user_input.strip().lower() == "exit":
            break
        pred, conf = model.predict(user_input, device=device)
        print(f"Prediction: {pred:.4f} (Confidence: {conf:.2f})") 