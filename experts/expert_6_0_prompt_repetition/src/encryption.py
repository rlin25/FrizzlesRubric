import boto3
import base64
import json
from typing import Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

KMS_KEY_ID = 'arn:aws:kms:us-east-2:804320821821:key/047b6705-f7ff-44c9-9296-c58a50c18574'  # Sir, replace with your actual KMS Key ID
kms_client = boto3.client('kms')

# Helper for AES encryption
BLOCK_SIZE = 128  # bits

def aes_encrypt(plaintext: bytes, key: bytes) -> bytes:
    iv = os.urandom(16)
    padder = padding.PKCS7(BLOCK_SIZE).padder()
    padded_data = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ciphertext  # prepend IV for decryption

def aes_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    iv = ciphertext[:16]
    actual_ciphertext = ciphertext[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(BLOCK_SIZE).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return plaintext

def encrypt_data(data: Any) -> str:
    if not isinstance(data, (str, bytes)):
        data = json.dumps(data)
    if isinstance(data, str):
        data = data.encode('utf-8')
    # Generate a data key
    response = kms_client.generate_data_key(KeyId=KMS_KEY_ID, KeySpec='AES_256')
    plaintext_key = response['Plaintext']
    encrypted_key = response['CiphertextBlob']
    # Encrypt the data with AES
    encrypted_data = aes_encrypt(data, plaintext_key)
    # Return both encrypted data and encrypted key, base64-encoded and JSON-packed
    result = {
        'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
        'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8')
    }
    return json.dumps(result)

def decrypt_data(ciphertext: str) -> Any:
    # Parse the JSON-packed encrypted data
    obj = json.loads(ciphertext)
    encrypted_data = base64.b64decode(obj['encrypted_data'])
    encrypted_key = base64.b64decode(obj['encrypted_key'])
    # Decrypt the data key with KMS
    response = kms_client.decrypt(CiphertextBlob=encrypted_key)
    plaintext_key = response['Plaintext']
    # Decrypt the data with AES
    plaintext = aes_decrypt(encrypted_data, plaintext_key)
    try:
        return json.loads(plaintext)
    except json.JSONDecodeError:
        return plaintext.decode('utf-8')

# Run this command to configure AWS credentials
# aws configure