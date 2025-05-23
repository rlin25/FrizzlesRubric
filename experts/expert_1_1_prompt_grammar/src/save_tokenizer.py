from transformers import AutoTokenizer

# Load the tokenizer from the pretrained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the tokenizer to the local model directory (absolute path)
model_dir = "/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/models/grammar_model"
tokenizer.save_pretrained(model_dir)

print(f"Tokenizer for '{model_name}' saved to '{model_dir}'") 