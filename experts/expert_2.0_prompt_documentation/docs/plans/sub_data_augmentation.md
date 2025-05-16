# Expert 2.0: Documentation Evaluation — Data Augmentation

## Objective
Implement and apply data augmentation techniques to enrich the training dataset while preserving documentation quality.

## Augmentation Techniques

### 1. Back Translation
```python
def back_translate(
    text: str,
    source_lang: str = 'en',
    target_lang: str = 'fr',
    model_name: str = 'Helsinki-NLP/opus-mt-en-fr'
) -> str:
    """
    Translate text to target language and back to source.
    Args:
        text: Input text to augment
        source_lang: Source language code
        target_lang: Target language code
        model_name: HuggingFace model name for translation
    Returns:
        Augmented text
    """
    translator = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Translate to target language
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = translator.generate(**inputs)
    target_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # Translate back to source
    reverse_model = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{target_lang}-{source_lang}")
    reverse_tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{target_lang}-{source_lang}")
    
    inputs = reverse_tokenizer(target_text, return_tensors="pt", padding=True)
    back_translated = reverse_model.generate(**inputs)
    return reverse_tokenizer.decode(back_translated[0], skip_special_tokens=True)
```

### 2. Synonym Replacement
```python
def replace_synonyms(
    text: str,
    nlp,
    max_replacements: int = 3
) -> str:
    """
    Replace words with their synonyms using spaCy.
    Args:
        text: Input text to augment
        nlp: spaCy language model
        max_replacements: Maximum number of words to replace
    Returns:
        Augmented text
    """
    doc = nlp(text)
    words_to_replace = random.sample(
        [token for token in doc if not token.is_stop and not token.is_punct],
        min(max_replacements, len(doc))
    )
    
    augmented = text
    for word in words_to_replace:
        if word.vector_norm:
            similar_words = [w.text for w in word.vocab if w.is_lower == word.is_lower
                           and w.prob >= -15 and w.text != word.text]
            if similar_words:
                replacement = random.choice(similar_words)
                augmented = augmented.replace(word.text, replacement)
    
    return augmented
```

### 3. Sentence Reordering
```python
def reorder_sentences(
    text: str,
    nlp,
    max_reorderings: int = 2
) -> str:
    """
    Reorder independent sentences while preserving meaning.
    Args:
        text: Input text to augment
        nlp: spaCy language model
        max_reorderings: Maximum number of reorderings to perform
    Returns:
        Augmented text
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Identify independent sentences (no dependencies on other sentences)
    independent_sentences = []
    for i, sent in enumerate(doc.sents):
        if not any(token.dep_ == "ref" for token in sent):
            independent_sentences.append(i)
    
    if len(independent_sentences) <= 1:
        return text
        
    # Perform random reorderings
    for _ in range(max_reorderings):
        if len(independent_sentences) >= 2:
            i, j = random.sample(independent_sentences, 2)
            sentences[i], sentences[j] = sentences[j], sentences[i]
    
    return " ".join(sentences)
```

### 4. Paraphrasing
```python
def paraphrase_text(
    text: str,
    model_name: str = "tuner007/pegasus_paraphrase"
) -> str:
    """
    Paraphrase text using a pre-trained model.
    Args:
        text: Input text to augment
        model_name: HuggingFace model name for paraphrasing
    Returns:
        Augmented text
    """
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_length=60,
        num_beams=10,
        num_return_sequences=1,
        temperature=1.5
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Quality Control

### 1. Documentation Quality Check
```python
def check_documentation_quality(
    original: str,
    augmented: str,
    nlp
) -> bool:
    """
    Verify that augmentation preserved documentation quality.
    Args:
        original: Original text
        augmented: Augmented text
        nlp: spaCy language model
    Returns:
        bool indicating if quality is preserved
    """
    # Check for key documentation elements
    original_doc = nlp(original)
    augmented_doc = nlp(augmented)
    
    # Compare number of sentences
    if len(list(original_doc.sents)) != len(list(augmented_doc.sents)):
        return False
    
    # Compare presence of key terms
    key_terms = ["function", "class", "method", "parameter", "return", "example"]
    original_terms = set(term.lower() for term in key_terms if term in original.lower())
    augmented_terms = set(term.lower() for term in key_terms if term in augmented.lower())
    
    if len(original_terms - augmented_terms) > 1:
        return False
    
    return True
```

### 2. Augmentation Pipeline
```python
def augment_dataset(
    df: pd.DataFrame,
    nlp,
    augmentation_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply augmentation techniques to dataset.
    Args:
        df: Input dataset
        nlp: spaCy language model
        augmentation_config: Configuration for augmentation techniques
    Returns:
        Augmented dataset
    """
    augmented_data = []
    
    for _, row in df.iterrows():
        text = row['prompt']
        label = row['label']
        
        # Apply each augmentation technique
        if augmentation_config.get('back_translation', False):
            augmented = back_translate(text)
            if check_documentation_quality(text, augmented, nlp):
                augmented_data.append({'prompt': augmented, 'label': label})
        
        if augmentation_config.get('synonym_replacement', False):
            augmented = replace_synonyms(text, nlp)
            if check_documentation_quality(text, augmented, nlp):
                augmented_data.append({'prompt': augmented, 'label': label})
        
        if augmentation_config.get('sentence_reordering', False):
            augmented = reorder_sentences(text, nlp)
            if check_documentation_quality(text, augmented, nlp):
                augmented_data.append({'prompt': augmented, 'label': label})
        
        if augmentation_config.get('paraphrasing', False):
            augmented = paraphrase_text(text)
            if check_documentation_quality(text, augmented, nlp):
                augmented_data.append({'prompt': augmented, 'label': label})
    
    return pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)
```

## Configuration
```python
AUGMENTATION_CONFIG = {
    'back_translation': True,
    'synonym_replacement': True,
    'sentence_reordering': True,
    'paraphrasing': True,
    'max_augmentations_per_text': 2,
    'quality_threshold': 0.8
}
```

## Performance Optimization
1. Batch processing for translations
2. Caching of translation models
3. Parallel processing for multiple augmentations
4. Early stopping for quality checks
5. Memory-efficient processing

## Testing Requirements
1. Unit tests for each augmentation technique
2. Quality preservation tests
3. Performance benchmarks
4. Edge case handling
5. Integration tests with dataset 