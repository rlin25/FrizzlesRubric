# Model Architecture

## Base Model
- Model: DistilBERT-base-uncased
- Vocabulary Size: 30,522 tokens
- Hidden Size: 768
- Number of Layers: 6
- Number of Attention Heads: 12
- Maximum Sequence Length: 512 tokens

## Classification Head
```python
nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
```

## Task Length Analysis
- Minimum words: 5
- Maximum words: 50
- Returns 0 if outside range, 1 if within range

## Scope Indicators
Limited set of indicators for task scope:

### Large Scope (0)
- implement
- create
- design
- develop
- build

### Specific (1)
- add
- fix
- update
- modify
- change

## Model Configuration
```python
{
    "model_type": "distilbert",
    "hidden_size": 768,
    "num_hidden_layers": 6,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02
}
```

## Forward Pass
1. Tokenize input text
2. Pass through DistilBERT
3. Use [CLS] token representation
4. Pass through classification head
5. Return prediction and confidence

## Prediction Method
```python
def predict(self, text: str) -> Tuple[float, float]:
    # Returns (prediction, confidence)
    # prediction: 0 or 1
    # confidence: 0.0 to 1.0
``` 