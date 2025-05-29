# Expert 2.0: Documentation Evaluation — Binary Classification Implementation

## Objective
Implement a binary classifier that evaluates documentation quality in Cursor AI prompts, with a focus on clear step delineation and comprehensive context.

## Technical Requirements
1. Input: Natural language prompt string
2. Output: Binary value (0 or 1)
3. Model: DistilBERT with classification head
4. Framework: PyTorch
5. Classification Threshold: 0.75 (optimized for clear step detection)

## Dependencies
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.12.0
- scikit-learn >= 1.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- spacy >= 3.5.0
- sentencepiece >= 0.1.99
- protobuf >= 3.20.0
- googletrans >= 3.1.0a0
- nltk >= 3.8.1
- tqdm >= 4.65.0

## Implementation Steps

### 1. Data Preprocessing
```python
def preprocess_prompt(prompt: str) -> str:
    # Remove special characters
    # Normalize whitespace
    # Convert to lowercase
    # Truncate to max length (512 tokens for DistilBERT)
    return processed_prompt
```

### 2. Model Architecture
```python
class DocumentationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[0][:, 0]
        logits = self.classifier(pooled)
        return self.sigmoid(logits)
```

### 3. Training Configuration
```python
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 2e-5,
    'epochs': 10,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'label_smoothing': 0.9
}
```

### 4. Classification Criteria
- Score > 0.75: Well-documented (1)
  - Clear step-by-step instructions
  - Comprehensive context
  - Proper task breakdown
- Score ≤ 0.75: Poorly documented (0)
  - Missing steps
  - Vague instructions
  - Insufficient context

### 5. Evaluation Metrics
```python
def compute_metrics(predictions, targets):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

## Integration Instructions
1. Load model weights from saved checkpoint
2. Implement inference pipeline
3. Add error handling for malformed inputs
4. Add logging for model predictions
5. Implement caching for frequently seen prompts

## Performance Requirements
- Inference time < 100ms per prompt
- Memory usage < 1GB
- Support for batch processing
- GPU acceleration optional

## Error Handling
```python
class DocumentationClassifierError(Exception):
    pass

def validate_input(prompt: str):
    if not isinstance(prompt, str):
        raise DocumentationClassifierError("Input must be a string")
    if len(prompt) == 0:
        raise DocumentationClassifierError("Input cannot be empty")
    if len(prompt) > 10000:  # Arbitrary limit
        raise DocumentationClassifierError("Input too long")
```

## Testing Requirements
1. Unit tests for preprocessing
2. Unit tests for model inference
3. Integration tests with sample prompts
4. Performance benchmarks
5. Error handling tests
6. Step detection validation

## Deployment Checklist
1. Model serialization
2. Input validation
3. Error handling
4. Logging setup
5. Performance monitoring
6. Model versioning
7. Threshold configuration
