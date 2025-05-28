# Ensemble System

## Architecture
- 3 DistilBERT models
- Independent training
- Confidence-based voting
- Model diversity through:
  - Different random seeds
  - Different dropout rates
  - Different learning rates

## Model Configuration
```python
ENSEMBLE_CONFIG = {
    "num_models": 3,
    "model_configs": [
        {
            "dropout": 0.1,
            "learning_rate": 2e-5
        },
        {
            "dropout": 0.15,
            "learning_rate": 1.8e-5
        },
        {
            "dropout": 0.12,
            "learning_rate": 2.2e-5
        }
    ]
}
```

## Voting Mechanism
```python
def ensemble_predict(text: str) -> Tuple[float, float]:
    predictions = []
    confidences = []
    
    for model in models:
        pred, conf = model.predict(text)
        predictions.append(pred)
        confidences.append(conf)
    
    # Weighted voting based on confidence
    weighted_pred = sum(p * c for p, c in zip(predictions, confidences))
    avg_confidence = sum(confidences) / len(confidences)
    
    return (1 if weighted_pred > 0.5 else 0), avg_confidence
```

## Training Process
1. Train each model independently
2. Use different random seeds
3. Validate each model separately
4. Combine for final predictions

## Model Diversity
- Different initialization
- Different hyperparameters
- Different training order
- Different validation splits

## Confidence Threshold
- Internal monitoring only
- Log low confidence predictions
- No impact on final output

## Ensemble Management
- Save all models
- Load on demand
- Parallel inference
- Memory efficient 