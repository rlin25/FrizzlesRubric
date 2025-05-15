# Expert 1.1 — Prompt Grammar Checker

## Overview

This expert module evaluates the **grammatical correctness** of developer prompts intended for Cursor AI, classifying each prompt as either grammatically correct (`1`) or incorrect (`0`). It handles grammar, punctuation, spacing, and capitalization errors but does **not** perform corrections.

---

## Dataset

- **Primary Source:** Entire [JFLEG dataset](https://web.eecs.umich.edu/~jfleg/)
- **Data Composition:**
  - **Class 0 (Incorrect):** Original learner sentences (~1,511 samples)
  - **Class 1 (Correct):** Four human-corrected reference sentences per original (~6,044 samples)
- **Labeling Scheme:**  
  Binary labels, where:  
  - `0` = prompt contains grammatical errors  
  - `1` = prompt is grammatically correct and fluent
- **Balancing:**  
  The dataset is inherently imbalanced (~4:1 correct:incorrect).  
  To address this, we will either:  
  - Downsample the `1` class to match `0` during training, **or**  
  - Use class weights in the loss function for balanced training

- **Splits:**  
  - Train: 70%  
  - Validation: 15%  
  - Test: 15%  
  Splits will be constructed to avoid data leakage by keeping all references and their original sentence in the same subset.

---

## Model

- **Architecture:** DistilBERT-based binary classifier (fine-tuned)
- **Input:** Raw prompt text (string)
- **Output:** Binary classification score (`0` or `1`), with optional confidence score
- **Task:** Binary classification of grammatical correctness only (no correction)
- **Scope:** Evaluate all grammar, punctuation, spacing, and capitalization errors
- **Prompt Length:** No minimum length; all prompt sizes supported

---

## Training

- **Loss Function:** Binary Cross-Entropy with optional class weighting to balance classes
- **Metrics:** Accuracy, Precision, Recall, F1-score on validation and test sets
- **Evaluation:** Monitor validation F1 for model selection and early stopping

---

## Deployment & Integration

- **Folder:** `expert_1.1_prompt_grammar`
- **Environment:** Dedicated virtual environment recommended
- **API Interface:**  
  - Input: JSON payload containing prompt text  
  - Output: JSON with binary label and confidence score  
- **Performance:** Target near real-time inference (<1s per prompt)
- **Logging:** Track predictions, confidence, timestamps, and version info for debugging

---

## Edge Cases & Handling

- Empty strings will be handled gracefully, returning a neutral or “correct” score by default.
- Code snippets mixed in prompts are passed through verbatim; grammar scoring focuses on natural language portions.
- Non-English text or extremely noisy input may produce low confidence scores.
- Spelling mistakes are considered part of grammar errors and lower scores.

---

## Next Steps

- Implement data processing script to convert JFLEG into the labeled format
- Prepare train/validation/test splits
- Set up DistilBERT fine-tuning pipeline with class weighting
- Develop inference API and integration tests
