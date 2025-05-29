# INCOMING BRANCH VERSION

# Expert 2.0: Documentation Evaluation — Cursor AI-Oriented Instructions

## Objective  
Train a binary classifier to determine if a natural language prompt for Cursor AI is **well-documented (1)** or **poorly documented (0)**, based on the inclusion of contextual background and explicit, broken-down instructions.

## Current Implementation Status
- Model: DistilBERT with classification head
- Classification Threshold: 0.75
- Framework: PyTorch
- Focus: Clear step delineation and comprehensive context
- Key Dependencies:
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

## Step 1: Define Binary Classification Task
Transform the classification problem into a binary label prediction:
- Output `1` if the input prompt includes background context and is broken down into subtasks.
- Output `0` if context is missing and/or the prompt is vague, high-level, or improperly scoped.

### Classification Criteria
- Score > 0.75: Well-documented (1)
  - Clear step-by-step instructions
  - Comprehensive context
  - Proper task breakdown
- Score ≤ 0.75: Poorly documented (0)
  - Missing steps
  - Vague instructions
  - Insufficient context

### Cursor AI Instruction:
> Implement a classifier model that takes in a developer-written natural language instruction and outputs a binary value indicating whether the prompt is well-documented.

## Step 2: Use Existing Prompt Data from Gemini
Use the dataset already generated via Gemini and stored in `\\data\\original_gemini`. This dataset contains natural language developer prompts that are:
- Thorough
- Multi-sentence
- Well-documented with contextual background and step-by-step breakdowns

All prompts in this dataset should be labeled as `1`.

### Cursor AI Instruction:
> Load the prompts from `\\data\\original_gemini` and treat all entries as positive examples (label `1`) for training.

## Step 3: Expand Dataset with Augmentation (Retain Label 1)
To enrich the positive examples and increase model generalizability:
- Apply text augmentation techniques that preserve documentation quality.
- Ensure all augmentations result in prompts that remain well-documented.

### Recommended Augmentation Techniques:
- **Back Translation**: Translate to another language and back to English to introduce variation while retaining meaning.
- **Synonym Replacement**: Replace non-critical terms with synonyms using contextual embeddings.
- **Sentence Reordering**: Reorder independent subtasks where order does not change meaning.
- **Paraphrasing**: Rephrase steps using language models to introduce structural diversity.

### Cursor AI Instruction:
> Augment the dataset by applying methods like back translation, synonym substitution, or paraphrasing to original prompts. Ensure that documentation quality is preserved. Label all augmented prompts with `1`.

## Step 4: Train Model with Robust Techniques
Use DistilBERT as a lightweight transformer model:
- Append a classification head with a sigmoid output
- Use binary cross-entropy loss
- Apply label smoothing (e.g., 0.9 for positives)
- Optionally integrate Focal Loss to reduce overconfidence
- Early stop based on validation F1 score

### Cursor AI Instruction:
> Fine-tune a DistilBERT model on the binary labeled dataset using a classification head. Apply label smoothing during training. Validate using a reserved portion of the dataset and monitor accuracy and F1 score.

## Step 5: Integration in Mixture of Experts (MoE)
This expert evaluates **documentation only**. Do not re-evaluate clarity or correctness:
- Return a binary value that reflects documentation strength only
- Feed this output into the MoE decision layer as one metric

### Cursor AI Instruction:
> Package the trained classifier into a callable service or module. When evaluating a prompt, return only a 0 or 1 that reflects documentation quality, without considering clarity or grammar. Forward this result into the MoE aggregator.

## Step 6: Evaluation & Validation
- Use only a portion of the `\\data\\original_gemini` dataset for testing
- Do not use augmented prompts for evaluation
- Metrics: Accuracy, Precision, Recall, F1
- Manually review misclassified examples if possible

### Cursor AI Instruction:
> Split the dataset such that only prompts from `\\data\\original_gemini` are used for evaluation. Compute accuracy, precision, recall, and F1 on the test subset. Do not evaluate on any augmented prompts.
