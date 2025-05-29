# Data Preparation Plan

## Labeling Criteria
- A prompt is labeled as containing a file reference if it includes a string with a common software file extension (e.g., .py, .js, .json, .md, etc.).
- Exclude URLs, code snippets, and ambiguous terms.

## Balancing the Dataset
- Downsample the majority class to match the number of examples in the minority class.
- Ensure both classes are equally represented.

## Train/Validation Split
- Split the labeled dataset into training and validation sets (e.g., 80/20 split).

## Data Augmentation (Optional)
- Consider augmenting data by inserting/removing file references in prompts to improve generalization. 