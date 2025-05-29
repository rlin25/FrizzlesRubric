import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import logging
from typing import Dict, Tuple, List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GranularityClassifier(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.1):
        super().__init__()
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
        
        # Task length thresholds
        self.min_words = 5
        self.max_words = 50
        
        # Scope indicators (limited set)
        self.scope_indicators = {
            'large_scope': [
                'implement',  # Major feature implementation
                'create',     # New component creation
                'design',     # System design
                'develop',    # Full feature development
                'build'       # Complete system building
            ],
            'specific': [
                'add',        # Single feature addition
                'fix',        # Bug fix
                'update',     # Existing feature update
                'modify',     # Small change
                'change'      # Minor modification
            ]
        }
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on the correct device
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)
        
        # Add batch dimension if not present
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
            
        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            return self.classifier(pooled_output)
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            raise
    
    def predict(self, text: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float]:
        """Make a prediction with confidence score."""
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            outputs = self(inputs["input_ids"], inputs["attention_mask"])
            prediction = outputs.item()
            confidence = abs(prediction - 0.5) * 2
            
            # Log low confidence predictions
            if confidence < 0.8:
                logger.warning(f"Low confidence prediction ({confidence:.2f}) for: {text}")
            
            return prediction, confidence
    
    def analyze_task_length(self, text: str) -> int:
        """Analyze task length and return 0 if too short or too long."""
        words = len(text.split())
        if words < self.min_words or words > self.max_words:
            return 0
        return 1
    
    def extract_scope_indicators(self, text: str) -> Dict[str, int]:
        """Count occurrences of scope indicators in text."""
        text_lower = text.lower()
        counts = {
            'large_scope': sum(1 for word in self.scope_indicators['large_scope'] if word in text_lower),
            'specific': sum(1 for word in self.scope_indicators['specific'] if word in text_lower)
        }
        return counts 