# Subplan 12: Dependencies

## Description
Dependencies required for the system.

## List
- Python3 (>=3.8)
- sentence-transformers (e.g., >=2.2.2)
- boto3 (>=1.20.0)
- nltk (>=3.6.0)
- numpy (>=1.21.0)
- AWS KMS permissions
- DynamoDB tables: `Prompts`, `PromptChecks`
- IAM roles for Lambda/API

## Setup
```bash
pip install sentence-transformers boto3 nltk numpy
python -m nltk.downloader stopwords
``` 