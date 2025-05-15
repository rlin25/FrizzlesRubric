import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_model_on_live_example(model_path, example_text, tokenizer_name="bert-base-uncased", device='cuda'):
    """Test a trained model on a live example and print the prediction."""

    # Load the trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the input text
    inputs = tokenizer(
        example_text, 
        padding=True, 
        truncation=True, 
        max_length=256, 
        return_tensors="pt"
    )

    # Move the input tensors to the specified device (GPU/CPU)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for efficiency
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Convert logits to predicted class
    predicted_class = torch.argmax(logits, dim=-1).cpu().numpy()

    # Print the result
    print(f"Input Text: {example_text}")
    print(f"Predicted Class: {predicted_class[0]}")

    return predicted_class[0]

if __name__ == "__main__":
    # Model and tokenizer settings
    model_path = './models/prompt_clarity_model'  # Path to the trained model
    tokenizer_name = "bert-base-uncased"  # Or use a different tokenizer if applicable
    device = 'cpu'  # Use 'cuda' if you have a GPU available

    # Example text(s) to test
    example_texts = [
        "Fix this code.",
        "Create a RESTful API in Flask that accepts user input and stores it in a PostgreSQL database.",
        "Correct this stupid thing. I've been trying to fix it for hours. I don't know what to do.",
        "Scrape the NBA website into a CSV file via BeautifulSoup.",
        "Fix code this.",
        "Create a API RESTful in Flask that user input accepts and itstores in a databasePostgreSQL.",
        "Correct this stupid thing. trying trying I've been to for fix it hours. I whatdon't know  to do.",
        "Scrape the website intoNBA  a  BeautifulSoupfile CSV via.",
    ]


    # Run the model on each example text
    for text in example_texts:
        predicted_class = test_model_on_live_example(model_path, text, tokenizer_name, device)
