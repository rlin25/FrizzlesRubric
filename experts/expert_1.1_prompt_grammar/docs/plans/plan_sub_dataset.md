# Dataset Plan

## Overview
This document details the dataset specifications for the Expert 1.1 Prompt Grammar Checker.

## Dataset Specifications

- **Primary Source:** Entire [JFLEG dataset](https://web.eecs.umich.edu/~jfleg/)
- **Data Composition:**
  - **Class 0 (Incorrect):** Original learner sentences (~1,511 samples)
  - **Class 1 (Correct):** Four human-corrected reference sentences per original (~6,044 samples)

## Labeling Scheme
Binary labels, where:  
- `0` = prompt contains grammatical errors  
- `1` = prompt is grammatically correct and fluent

## Data Balancing
The dataset is inherently imbalanced (~4:1 correct:incorrect).  
To address this, we will either:  
- Downsample the `1` class to match `0` during training, **or**  
- Use class weights in the loss function for balanced training

## Data Splits
- Train: 70%  
- Validation: 15%  
- Test: 15%  

**Important Note:** Splits will be constructed to avoid data leakage by keeping all references and their original sentence in the same subset.

## Implementation Tasks
1. Download and process JFLEG dataset
2. Implement data balancing strategy
3. Create train/validation/test splits
4. Develop data loading pipeline
5. Implement data augmentation if needed 