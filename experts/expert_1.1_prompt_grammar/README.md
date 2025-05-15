# Expert 1.1: Prompt Grammar Checker

A binary classifier that evaluates the grammatical correctness of developer prompts for Cursor AI.

## Overview

This module evaluates the grammatical correctness of developer prompts, classifying each prompt as either grammatically correct (`1`) or incorrect (`0`). It handles grammar, punctuation, spacing, and capitalization errors but does not perform corrections.

## Features

- Binary classification of grammatical correctness
- Handles all prompt sizes
- Processes grammar, punctuation, spacing, and capitalization
- Near real-time inference (<1s per prompt)
- Confidence scoring for predictions

## Project Structure

- `train.py`: Training script for the DistilBERT-based classifier
- `data_processing.py`: JFLEG dataset processing and preparation
- `inference_api.py`: FastAPI-based inference service
- `docs/`: Documentation and planning
- `requirements.txt`: Project dependencies

## Dataset

The model is trained on the JFLEG dataset:
- Class 0 (Incorrect): Original learner sentences (~1,511 samples)
- Class 1 (Correct): Four human-corrected reference sentences per original (~6,044 samples)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Processing:
```bash
python data_processing.py
```

2. Training:
```bash
python train.py
```

3. Running the API:
```bash
python inference_api.py
```

## API Interface

The inference API accepts JSON payloads with the following structure:
```json
{
    "prompt": "Your prompt text here"
}
```

And returns:
```json
{
    "is_correct": 1,
    "confidence": 0.95
}
```

## Edge Cases

- Empty strings return a neutral/correct score
- Code snippets are passed through verbatim
- Non-English text may produce low confidence scores
- Spelling mistakes are considered grammar errors

## Contributing

Please refer to the contribution guidelines in the `docs` folder for information on how to contribute to this project. 