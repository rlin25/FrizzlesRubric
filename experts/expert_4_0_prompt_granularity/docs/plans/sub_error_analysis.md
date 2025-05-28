# Error Analysis

## Error Logging
```python
class ErrorLog:
    def __init__(self):
        self.errors = []
        
    def log_error(self, prompt: str, predicted: int, true: int, confidence: float):
        self.errors.append({
            'prompt': prompt,
            'predicted': predicted,
            'true': true,
            'confidence': confidence,
            'length': len(prompt.split()),
            'keywords': self.extract_keywords(prompt)
        })
```

## Pattern Analysis
```python
def analyze_patterns(errors: List[Dict]) -> Dict:
    df = pd.DataFrame(errors)
    return {
        'length_distribution': df['length'].describe(),
        'confidence_distribution': df['confidence'].describe(),
        'keyword_frequency': df['keywords'].value_counts(),
        'error_by_length': df.groupby('length')['predicted'].mean(),
        'error_by_confidence': df.groupby(pd.qcut(df['confidence'], 5))['predicted'].mean()
    }
```

## Performance Metrics
- Error rate by length
- Error rate by confidence
- Error rate by keywords
- Confusion matrix
- Error patterns

## Analysis Categories
1. Length-based analysis
   - Too short prompts
   - Too long prompts
   - Optimal length range

2. Keyword-based analysis
   - Missing keywords
   - Ambiguous keywords
   - Keyword combinations

3. Confidence-based analysis
   - Low confidence errors
   - High confidence errors
   - Confidence distribution

## Error Types
1. False Positives (0->1)
   - Large scope tasks classified as specific
   - Common patterns
   - Mitigation strategies

2. False Negatives (1->0)
   - Specific tasks classified as large scope
   - Common patterns
   - Mitigation strategies

## Reporting
- Error summary
- Pattern identification
- Improvement suggestions
- Model adjustment recommendations 