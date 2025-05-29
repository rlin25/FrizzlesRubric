import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DistilBertFileClassifier(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits

    def predict(self, text, device='cpu'):
        self.eval()
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            prob = torch.sigmoid(logits).item()
            label = int(prob >= 0.5)
        return label, prob 