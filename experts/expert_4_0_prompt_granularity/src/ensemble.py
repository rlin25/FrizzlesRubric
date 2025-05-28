import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
from .model import GranularityClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GranularityEnsemble:
    def __init__(
        self,
        models: List[GranularityClassifier],
        confidence_threshold: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.models = models
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()
            model.to(device)
    
    def predict(self, text: str) -> Tuple[float, float, Dict[str, float]]:
        """
        Make ensemble prediction with confidence scores and individual model predictions.
        
        Returns:
            Tuple containing:
            - Final prediction (0 or 1)
            - Ensemble confidence score
            - Dictionary of individual model predictions and confidences
        """
        predictions = []
        confidences = []
        individual_results = {}
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            pred, conf = model.predict(text, self.device)
            predictions.append(pred)
            confidences.append(conf)
            individual_results[f'model_{i}'] = {
                'prediction': pred,
                'confidence': conf
            }
        
        # Calculate ensemble prediction (weighted average)
        ensemble_pred = np.average(predictions, weights=confidences)
        ensemble_conf = abs(ensemble_pred - 0.5) * 2
        
        # Final binary prediction
        final_pred = 1 if ensemble_pred > 0.5 else 0
        
        # Log low confidence predictions
        if ensemble_conf < self.confidence_threshold:
            logger.warning(
                f"Low ensemble confidence ({ensemble_conf:.2f}) for: {text}\n"
                f"Individual confidences: {confidences}"
            )
        
        return final_pred, ensemble_conf, individual_results
    
    def analyze_disagreement(self, text: str) -> Dict[str, float]:
        """
        Analyze disagreement among models for a given text.
        
        Returns:
            Dictionary containing:
            - disagreement_score: Measure of model disagreement (0-1)
            - std_dev: Standard deviation of predictions
            - max_diff: Maximum difference between any two predictions
        """
        predictions = []
        
        for model in self.models:
            pred, _ = model.predict(text, self.device)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        return {
            'disagreement_score': float(np.std(predictions)),
            'std_dev': float(np.std(predictions)),
            'max_diff': float(np.max(predictions) - np.min(predictions))
        }
    
    def get_model_agreement(self, text: str) -> Dict[str, int]:
        """
        Get the number of models agreeing with the ensemble prediction.
        
        Returns:
            Dictionary containing:
            - agreement_count: Number of models agreeing with ensemble
            - total_models: Total number of models
            - agreement_ratio: Ratio of agreeing models
        """
        final_pred, _, individual_results = self.predict(text)
        
        agreement_count = sum(
            1 for result in individual_results.values()
            if (result['prediction'] > 0.5) == final_pred
        )
        
        return {
            'agreement_count': agreement_count,
            'total_models': len(self.models),
            'agreement_ratio': agreement_count / len(self.models)
        }
    
    def save_ensemble(self, base_path: str):
        """Save all models in the ensemble."""
        for i, model in enumerate(self.models):
            model_path = f"{base_path}_model_{i}.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model {i} to {model_path}")
    
    @classmethod
    def load_ensemble(cls, model_paths: List[str], **kwargs) -> 'GranularityEnsemble':
        """Load an ensemble from saved model paths."""
        models = []
        for path in model_paths:
            model = GranularityClassifier()
            model.load_state_dict(torch.load(path))
            models.append(model)
        
        return cls(models=models, **kwargs) 