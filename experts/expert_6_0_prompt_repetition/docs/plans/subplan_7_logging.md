# Subplan 7: Logging

## Description
Log every similarity check and its result in DynamoDB.

## Table: PromptChecks
- Attributes:
  - `check_id` (UUID)
  - `timestamp` (ISO)
  - `encrypted_prompt` (KMS-encrypted)
  - `result` (0 or 1)
  - `similarity_score` (float, if repeat)
  - `most_similar_prompt_id` (if repeat)
  - `threshold` (float)
- Consider secondary indexes for analytics (e.g., by timestamp)

## Operations
- Log every similarity check and result
- Handle DynamoDB errors (provisioned throughput, missing keys, etc.)
- Integrate logging with API endpoint

## Pseudocode
```python
import uuid, time

def log_check(raw_prompt, result, similarity_score, most_similar_prompt_id, threshold=0.85):
    encrypted_prompt = encrypt_data(raw_prompt)
    item = {
        'check_id': str(uuid.uuid4()),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'encrypted_prompt': encrypted_prompt,
        'result': result,
        'similarity_score': similarity_score,
        'most_similar_prompt_id': most_similar_prompt_id,
        'threshold': threshold
    }
    log_table = dynamodb.Table('PromptChecks')
    log_table.put_item(Item=item)
``` 