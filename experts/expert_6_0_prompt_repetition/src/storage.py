import boto3
import hashlib
import time
from typing import List, Dict

# Only store raw prompts in DynamoDB

dynamodb = boto3.resource('dynamodb')
PROMPTS_TABLE = 'PromptChecks'

def get_prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

def store_prompt(prompt: str):
    table = dynamodb.Table(PROMPTS_TABLE)
    prompt_id = get_prompt_id(prompt)
    item = {
        'id': prompt_id,
        'prompt': prompt,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    print("DEBUG: Storing item in DynamoDB:", item)
    table.put_item(Item=item)

def get_all_prompts() -> List[Dict]:
    table = dynamodb.Table(PROMPTS_TABLE)
    response = table.scan()
    return response.get('Items', []) 