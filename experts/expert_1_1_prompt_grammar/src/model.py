import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptGrammarClassifier(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.1):
        """
        Initialize the Prompt Grammar Classifier.
        
        Args:
            model_name (str): Name of the pre-trained DistilBERT model
            dropout (float): Dropout rate for the classification head
        """
        super().__init__()
        logger.info(f"Initializing model with {model_name}")
        
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        logger.info("Model initialized successfully")
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Binary classification scores
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return self.classifier(pooled_output)
    
    def predict(self, text: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float]:
        """
        Make a prediction for a single text input.
        
        Args:
            text (str): Input text to classify
            device (str): Device to run inference on
            
        Returns:
            Tuple[float, float]: (prediction, confidence)
        """
        self.eval()
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            # Get prediction
            outputs = self(inputs["input_ids"], inputs["attention_mask"])
            prediction = outputs.item()
            
            # Calculate confidence (distance from decision boundary)
            confidence = abs(prediction - 0.5) * 2
            
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
        model.tokenizer = DistilBertTokenizer.from_pretrained(path)
        logger.info("Model loaded successfully")
        return model 