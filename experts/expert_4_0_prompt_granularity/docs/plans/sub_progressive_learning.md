# Progressive Learning

## Two-Phase Training
1. Clear Examples Phase
2. Full Dataset Phase

## Clear Example Filtering
```python
def filter_clear_examples(data: List[Dict]) -> List[Dict]:
    clear_examples = []
    for example in data:
        if is_clear_example(example):
            clear_examples.append(example)
    return clear_examples

def is_clear_example(example: Dict) -> bool:
    # Check length
    if not (5 <= len(example['prompt'].split()) <= 50):
        return False
        
    # Check keywords
    keywords = extract_keywords(example['prompt'])
    if not has_clear_keywords(keywords):
        return False
        
    # Check label consistency
    if not is_label_consistent(example):
        return False
        
    return True
```

## Training Phases
### Phase 1: Clear Examples
- Train on filtered clear examples
- Higher learning rate
- Fewer epochs
- Focus on basic patterns

### Phase 2: Full Dataset
- Fine-tune on all examples
- Lower learning rate
- More epochs
- Focus on edge cases

## Configuration
```python
PROGRESSIVE_CONFIG = {
    "phase1": {
        "learning_rate": 3e-5,
        "epochs": 3,
        "batch_size": 32
    },
    "phase2": {
        "learning_rate": 1e-5,
        "epochs": 5,
        "batch_size": 32
    }
}
```

## Clear Example Criteria
1. Length requirements
   - Between 5 and 50 words
   - Clear sentence structure

2. Keyword requirements
   - Single clear scope indicator
   - No ambiguous terms
   - Consistent with label

3. Label consistency
   - Matches length analysis
   - Matches keyword analysis
   - No conflicting indicators

## Fine-tuning Strategy
1. Load Phase 1 model
2. Reduce learning rate
3. Train on full dataset
4. Use early stopping
5. Save best model

## Monitoring
- Phase 1 performance
- Phase 2 performance
- Improvement metrics
- Error patterns
- Learning curves 