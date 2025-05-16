from transformers import AutoModel, AutoTokenizer
import torch

def test_model():
    # Load model and tokenizer
    model_path = "experts/expert_1.1_prompt_grammar/models"
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    
    # Test prompt
    test_prompt = "Fix the grammar in this sentence: 'he go to the store yesterday'"
    print(f"\nTest prompt: {test_prompt}")
    
    # Tokenize input
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Print model outputs
    print("\nModel outputs shape:", outputs.last_hidden_state.shape)
    print("Model outputs:", outputs.last_hidden_state[0, 0, :5])  # Print first 5 values of first token

if __name__ == "__main__":
    test_model() 