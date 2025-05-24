import requests
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

API_URL = "http://localhost:8080/expert1/predict"
CHECKPOINT_PATH = "/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/models/grammar_model/checkpoint-987"

def query_api(text):
    response = requests.post(API_URL, json={"input": text})
    if response.status_code == 200:
        return response.json()["prediction"], response.json()["confidence"]
    else:
        return None, None

def query_checkpoint(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model.to(device)
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0].item()
        prediction = torch.argmax(probs, dim=1).item()
    return prediction, confidence

test_texts = [
    "Fix this code.",
    "Create a RESTful API in Flask that accepts user input and stores it in a PostgreSQL database.",
    "Correct this stupid thing. I've been trying to fix it for hours. I don't know what to do.",
    "Scrape the NBA website into a CSV file via BeautifulSoup.",
    "Fix code this.",
    "Create a API RESTful in Flask that user input accepts and itstores in a databasePostgreSQL.",
    "Correct this stupid thing. trying trying I've been to for fix it hours. I whatdon't know to do.",
    "Scrape the website intoNBA a BeautifulSoupfile CSV via."
]

print("Comparing API vs. Checkpoint-987 predictions:\n")
for text in test_texts:
    api_pred, api_conf = query_api(text)
    chk_pred, chk_conf = query_checkpoint(text)
    print(f"Input: {text}")
    print(f"  API:         {'Correct' if api_pred == 1 else 'Incorrect'} (conf: {api_conf})")
    print(f"  Checkpoint:  {'Correct' if chk_pred == 1 else 'Incorrect'} (conf: {chk_conf})")
    print(f"  Match:       {api_pred == chk_pred}")
    print("-") 