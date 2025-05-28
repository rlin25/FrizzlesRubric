import boto3
import uuid
import time
from decimal import Decimal

PROMPT_CHECKS_TABLE = 'PromptChecks'
dynamodb = boto3.resource('dynamodb')

def log_prompt_check(encrypted_prompt: str, result: int, similarity_score: float = None, most_similar_prompt_id: str = None, threshold: float = 0.85):
    table = dynamodb.Table(PROMPT_CHECKS_TABLE)
    item = {
        'id': str(uuid.uuid4()),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'encrypted_prompt': encrypted_prompt,
        'result': result,
        'threshold': Decimal(str(threshold))
    }
    if similarity_score is not None:
        item['similarity_score'] = Decimal(str(similarity_score))
    if most_similar_prompt_id is not None:
        item['most_similar_prompt_id'] = most_similar_prompt_id
    table.put_item(Item=item) 