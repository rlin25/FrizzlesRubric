# Subplan 3: Encryption

## Description
Encrypt the preprocessed prompt and its embedding using AWS KMS before storing in DynamoDB. Decrypt when reading for similarity checks.

## Steps
- Use AWS KMS to encrypt preprocessed prompt (string) and embedding (list of floats, JSON-serialized)
- Use a specific KMS key (configurable by ARN or alias)
- Use boto3's KMS client for encrypt/decrypt
- Store ciphertext as base64-encoded string
- Decrypt ciphertext when reading for similarity checks
- Handle KMS errors (permissions, throttling, etc.)

## Input
- Preprocessed prompt (string)
- Embedding (list of floats)

## Output
- Encrypted prompt (base64 string)
- Encrypted embedding (base64 string, JSON-serialized list)

## Pseudocode
```python
import boto3, base64, json
kms = boto3.client('kms')
KMS_KEY_ID = 'alias/your-key-alias'

def encrypt_data(data):
    if not isinstance(data, bytes):
        data = json.dumps(data).encode() if not isinstance(data, str) else data.encode()
    response = kms.encrypt(KeyId=KMS_KEY_ID, Plaintext=data)
    return base64.b64encode(response['CiphertextBlob']).decode()

def decrypt_data(ciphertext_b64):
    ciphertext = base64.b64decode(ciphertext_b64)
    response = kms.decrypt(CiphertextBlob=ciphertext)
    return response['Plaintext'].decode()
``` 