from flask import Flask, request, jsonify
from .preprocessing import preprocess_prompt
from .embedding import generate_embedding
from .encryption import encrypt_data, decrypt_data
from .storage import store_prompt, get_all_prompts, get_prompt_id
from .similarity import cosine_similarity
from .logging_utils import log_prompt_check
from .analytics import get_analytics
import numpy as np
import boto3
import signal
import atexit
import sys

app = Flask(__name__)
SIMILARITY_THRESHOLD = 0.85
MAX_PROMPT_LENGTH = 512

@app.route('/check_prompt', methods=['POST'])
def check_prompt():
    data = request.get_json()
    prompt = data.get('prompt', '')
    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({'error': 'Prompt exceeds length limit'}), 400
    preprocessed = preprocess_prompt(prompt)
    encrypted_prompt = encrypt_data(preprocessed)
    # Retrieve all stored prompts
    stored = get_all_prompts()
    most_similar = None
    max_sim = -1
    for item in stored:
        if 'prompt' not in item:
            continue
        sim = 1.0 if prompt == item['prompt'] else 0.0
        if sim > max_sim:
            max_sim = sim
            most_similar = item
    if max_sim > SIMILARITY_THRESHOLD:
        log_prompt_check(encrypted_prompt, 0, max_sim, most_similar['id'], SIMILARITY_THRESHOLD)
        return jsonify({'result': 0})
    else:
        store_prompt(prompt)  # Store only the raw prompt
        log_prompt_check(encrypted_prompt, 1, None, None, SIMILARITY_THRESHOLD)
        return jsonify({'result': 1})

@app.route('/analytics', methods=['GET'])
def analytics():
    return jsonify(get_analytics())

@app.route('/review', methods=['GET'])
def review():
    # For simplicity, just return all repeats
    from .logging_utils import dynamodb, PROMPT_CHECKS_TABLE
    table = dynamodb.Table(PROMPT_CHECKS_TABLE)
    response = table.scan()
    repeats = [item for item in response.get('Items', []) if item.get('result') == 0]
    return jsonify(repeats)

def clear_promptchecks_table():
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('PromptChecks')
    scan = table.scan(ProjectionExpression='id')
    with table.batch_writer() as batch:
        for item in scan.get('Items', []):
            batch.delete_item(Key={'id': item['id']})

def clear_prompts_table():
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('PromptChecks')  # If PROMPTS_TABLE is different, update here
    scan = table.scan(ProjectionExpression='id')
    with table.batch_writer() as batch:
        for item in scan.get('Items', []):
            batch.delete_item(Key={'id': item['id']})

def cleanup(*args, **kwargs):
    try:
        clear_promptchecks_table()
        clear_prompts_table()
        print("PromptChecks and PROMPTS_TABLE tables cleared on shutdown.")
    except Exception as e:
        print(f"Error clearing tables on shutdown: {e}")

# Register cleanup with atexit and signal handlers
atexit.register(cleanup)
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda signum, frame: (cleanup(), sys.exit(0)))

@app.route('/reset_promptchecks', methods=['POST'])
def reset_promptchecks():
    try:
        clear_promptchecks_table()
        return jsonify({'success': True, 'message': 'PromptChecks table cleared.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8007) 