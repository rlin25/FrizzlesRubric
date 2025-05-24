import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrammarClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        """
        Initialize the Prompt Grammar Classifier.
        
        Args:
            model_name (str): Name of the pre-trained DistilBERT model
            num_labels (int): Number of labels for the classification task
        """
        super().__init__()
        logger.info(f"Initializing model with {model_name}")
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info("Model initialized successfully")
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor, optional): Ground truth labels
            
        Returns:
            dict: Classification outputs or loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def predict(self, text: str) -> Tuple[int, float]:
        """
        Make a prediction for a single text input.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            Tuple[int, float]: (prediction, confidence)
        """
        self.eval()
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get prediction
            outputs = self(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            confidence = torch.max(probs, dim=1)[0].item()
            prediction = torch.argmax(probs, dim=1).item()
            
            return prediction, confidence
    
    def save(self, path: str):
        """Save the model and tokenizer."""
        logger.info(f"Saving model to {path}")
        torch.save(self.state_dict(), f"{path}/model.pt")
        self.tokenizer.save_pretrained(path)
        logger.info("Model saved successfully")
    
    @classmethod
    def load(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load a saved model."""
        logger.info(f"Loading model from {path}")
        model = cls()
        model.load_state_dict(torch.load(f"{path}/model.pt", map_location=device))
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info("Model loaded successfully")
        return model 

def create_model(model_name="distilbert-base-uncased"):
    """Create and return a new model instance."""
    return GrammarClassifier(model_name=model_name) 