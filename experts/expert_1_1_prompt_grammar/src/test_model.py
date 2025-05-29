import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/model_checkpoint"
TOKENIZER_NAME = "distilbert-base-uncased"  # Use the original pretrained model name

def main():
    print("Loading model and tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    print("\nEnter a prompt to classify (type 'exit' to quit):", flush=True)
    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() == 'exit':
                print("Exiting.")
                break
            if not prompt:
                continue
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()
                print(f"Result: {prediction}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main() 