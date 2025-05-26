# Model Architecture Plan

## Base Model
- Use DistilBERT as the transformer backbone for its efficiency and speed.

## Classification Head
- Add a single linear layer on top of DistilBERT's pooled output.
- Apply a sigmoid activation to produce a probability.

## Input/Output
- Input: Raw prompt text (max 256 tokens).
- Output: Hard label (0/1) indicating file reference presence, and a confidence score (probability). 