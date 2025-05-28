import boto3
import numpy as np

PROMPT_CHECKS_TABLE = 'PromptChecks'
dynamodb = boto3.resource('dynamodb')

def get_analytics():
    table = dynamodb.Table(PROMPT_CHECKS_TABLE)
    response = table.scan()
    items = response.get('Items', [])
    total = len(items)
    repeats = [item for item in items if item.get('result') == 0]
    num_repeats = len(repeats)
    percent_repeats = (num_repeats / total * 100) if total else 0
    sim_scores = [item.get('similarity_score', 0) for item in repeats if 'similarity_score' in item]
    avg_sim = float(np.mean(sim_scores)) if sim_scores else 0
    lengths = [len(item.get('encrypted_prompt', '')) for item in items]
    timestamps = [item.get('timestamp') for item in items]
    return {
        'total_checked': total,
        'num_repeats': num_repeats,
        'percent_repeats': percent_repeats,
        'avg_similarity_score': avg_sim,
        'prompt_lengths': lengths,
        'timestamps': timestamps
    } 