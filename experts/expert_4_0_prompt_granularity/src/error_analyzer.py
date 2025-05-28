import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    def __init__(self, model_outputs: Dict[str, List[float]], true_labels: List[int]):
        """
        Initialize error analyzer with model predictions and true labels.
        
        Args:
            model_outputs: Dictionary mapping model names to their predictions
            true_labels: List of true labels
        """
        self.model_outputs = model_outputs
        self.true_labels = np.array(true_labels)
        self.error_patterns = defaultdict(list)
        self.confusion_matrices = {}
        
    def analyze_errors(self, texts: List[str]) -> Dict[str, Dict]:
        """
        Analyze prediction errors across all models.
        
        Args:
            texts: List of input texts corresponding to predictions
            
        Returns:
            Dictionary containing error analysis results
        """
        results = {
            'error_rates': {},
            'error_patterns': {},
            'confusion_matrices': {},
            'error_examples': defaultdict(list)
        }
        
        # Calculate error rates and confusion matrices for each model
        for model_name, predictions in self.model_outputs.items():
            pred_array = np.array(predictions)
            error_rate = np.mean(pred_array != self.true_labels)
            results['error_rates'][model_name] = error_rate
            
            # Calculate confusion matrix
            cm = confusion_matrix(self.true_labels, pred_array)
            results['confusion_matrices'][model_name] = cm
            
            # Find error examples
            error_indices = np.where(pred_array != self.true_labels)[0]
            for idx in error_indices:
                results['error_examples'][model_name].append({
                    'text': texts[idx],
                    'predicted': int(pred_array[idx]),
                    'true': int(self.true_labels[idx])
                })
        
        # Analyze error patterns
        results['error_patterns'] = self._analyze_error_patterns(texts)
        
        return results
    
    def _analyze_error_patterns(self, texts: List[str]) -> Dict[str, List[str]]:
        """Analyze common patterns in misclassified examples."""
        patterns = {
            'false_positives': [],
            'false_negatives': []
        }
        
        for model_name, predictions in self.model_outputs.items():
            pred_array = np.array(predictions)
            
            # Find false positives
            fp_indices = np.where((pred_array == 1) & (self.true_labels == 0))[0]
            for idx in fp_indices:
                patterns['false_positives'].append(texts[idx])
            
            # Find false negatives
            fn_indices = np.where((pred_array == 0) & (self.true_labels == 1))[0]
            for idx in fn_indices:
                patterns['false_negatives'].append(texts[idx])
        
        return patterns
    
    def plot_confusion_matrices(self, save_dir: Optional[str] = None):
        """Plot confusion matrices for all models."""
        n_models = len(self.model_outputs)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, cm) in zip(axes, self.confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'confusion_matrices.png'
            plt.savefig(save_path)
            logger.info(f"Saved confusion matrices plot to {save_path}")
        
        plt.close()
    
    def generate_error_report(self, save_path: str):
        """Generate a comprehensive error analysis report."""
        report = {
            'error_rates': self.error_rates,
            'error_patterns': self.error_patterns,
            'confusion_matrices': {
                name: cm.tolist() for name, cm in self.confusion_matrices.items()
            }
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Error analysis report saved to {save_path}")
    
    def get_error_examples(self, model_name: str, error_type: str = 'all') -> List[Dict]:
        """
        Get examples of errors for a specific model and error type.
        
        Args:
            model_name: Name of the model
            error_type: Type of errors to retrieve ('fp', 'fn', or 'all')
            
        Returns:
            List of error examples with their details
        """
        if model_name not in self.model_outputs:
            raise ValueError(f"Model {model_name} not found")
        
        predictions = np.array(self.model_outputs[model_name])
        
        if error_type == 'fp':
            indices = np.where((predictions == 1) & (self.true_labels == 0))[0]
        elif error_type == 'fn':
            indices = np.where((predictions == 0) & (self.true_labels == 1))[0]
        else:  # 'all'
            indices = np.where(predictions != self.true_labels)[0]
        
        return [
            {
                'text': texts[idx],
                'predicted': int(predictions[idx]),
                'true': int(self.true_labels[idx])
            }
            for idx in indices
        ]
    
    def analyze_confidence_distribution(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze the distribution of confidence scores for correct and incorrect predictions.
        
        Returns:
            Dictionary containing confidence statistics for each model
        """
        confidence_stats = {}
        
        for model_name, predictions in self.model_outputs.items():
            pred_array = np.array(predictions)
            correct_mask = pred_array == self.true_labels
            
            confidence_stats[model_name] = {
                'correct_mean': float(np.mean(predictions[correct_mask])),
                'correct_std': float(np.std(predictions[correct_mask])),
                'incorrect_mean': float(np.mean(predictions[~correct_mask])),
                'incorrect_std': float(np.std(predictions[~correct_mask]))
            }
        
        return confidence_stats 