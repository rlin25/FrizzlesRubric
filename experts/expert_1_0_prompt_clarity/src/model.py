from transformers import AutoModelForSequenceClassification

def create_model(model_name="distilbert-base-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model
