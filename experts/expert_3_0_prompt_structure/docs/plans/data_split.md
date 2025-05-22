# Data Split Strategy

## Overview
The data is split into training, validation, and test sets with stratification by multiple factors to ensure balanced representation across all categories.

## Sample Counts
### Base Samples
- Positive samples: 1,007
- Negative samples: 574
- Total base samples: 1,581

### After Augmentation
- Additional samples per base sample: 0.8
- Total augmented samples: 1,265
- Total training samples: 2,846

## Split Ratios
- Training: 70%
- Validation: 15%
- Test: 15%

## Data Distribution

### Training Set (70%)
- Positive samples: 705
- Negative samples: 402
- Total: 1,107 samples

### Validation Set (15%)
- Positive samples: 151
- Negative samples: 86
- Total: 237 samples

### Test Set (15%)
- Positive samples: 151
- Negative samples: 86
- Total: 237 samples

## Stratification
### By Length
- Short prompts (< 50 words)
- Medium prompts (50-100 words)
- Long prompts (> 100 words)

### By Pattern Type
- Explicit file references
- Implicit file references
- Function references
- Component references
- Line number references

### By OS Type
- Windows paths
- Linux paths
- macOS paths
- Generic paths

### By Programming Language
- JavaScript/TypeScript
- Python
- Java
- C/C++
- Other languages

## Negative Sample Selection
### Criteria
1. Most diverse patterns
2. Most ambiguous cases
3. Most challenging examples

### Excluded Samples
- Total excluded: 0 samples (using all available data)
- Selection based on:
  - Pattern diversity
  - Complexity level
  - Edge cases

## Data Augmentation
### Techniques
1. Synonym Replacement
   - Probability: 0.4 (increased from 0.3)
   - Max replacements: 3 (increased from 2)
   - Expected additional samples: 1.2 per base sample

2. Pattern Variation
   - Probability: 0.3 (increased from 0.2)
   - Max variations: 2 (increased from 1)
   - Expected additional samples: 0.6 per base sample

3. Context Variation
   - Probability: 0.2
   - Max variations: 1
   - Expected additional samples: 0.2 per base sample

### Total Augmentation
- Additional samples per base sample: 2.0 (increased from 0.8)
- Total augmented samples: 3,162
- Final dataset size: 4,743 samples

## Cross Validation
- Folds: 5
- Stratification: true
- Shuffle: true
- Stratification by: length, pattern type, OS type, programming language

## Data Sources
### Positive Samples
- structure_positive_length_short.csv (129 samples)
- structure_positive_length_long.csv (426 samples)
- structure_positive_length_longest_system_linux.csv (149 samples)
- structure_positive_length_longest_system_macos.csv (149 samples)
- structure_positive_length_longest_system_windows.csv (154 samples)

### Negative Samples
- structure_negative_length_short.csv (149 samples)
- structure_negative_length_long.csv (425 samples) 