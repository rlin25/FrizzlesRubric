# Subplan 4: DynamoDB Storage

## Description
Store and retrieve encrypted prompts and embeddings in DynamoDB.

## Table: Prompts
- Primary key: `prompt_id` (SHA256 hash of preprocessed prompt)
- Attributes:
  - `prompt_id` (string)
  - `encrypted_prompt` (string, KMS-encrypted)
  - `encrypted_embedding` (string, KMS-encrypted, JSON list)
  - `created_at` (ISO timestamp)
- Consider secondary indexes for analytics (e.g., by created_at)

## Operations
- Store new prompt and embedding if unique
- Retrieve all stored embeddings for similarity check
- Handle DynamoDB errors (provisioned throughput, missing keys, etc.)

## Pseudocode
```python
import boto3
import hashlib
import time
import uuid
import json

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Prompts')

def store_prompt(preprocessed_prompt, encrypted_prompt, encrypted_embedding):
    prompt_id = hashlib.sha256(preprocessed_prompt.encode()).hexdigest()
    item = {
        'prompt_id': prompt_id,
        'encrypted_prompt': encrypted_prompt,
        'encrypted_embedding': encrypted_embedding,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ')
    }
    table.put_item(Item=item)
    return prompt_id

def get_all_embeddings():
    response = table.scan(ProjectionExpression='prompt_id, encrypted_embedding')
    return response['Items'] 