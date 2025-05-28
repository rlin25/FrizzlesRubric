# Subplan 8: Analytics

## Description
Compute and expose analytics based on prompt checks.

## Steps
- Compute total prompts checked (count rows in PromptChecks)
- Compute number and percentage of repeats flagged (result == 0)
- Compute average similarity score for repeats
- Compute distribution of prompt lengths (from decrypted prompts)
- Compute timestamps of checks (for time-based analytics)
- Use data from `PromptChecks` table
- Handle DynamoDB scan errors and missing data

## Output
- Analytics summary (JSON)

## Pseudocode
```python
def compute_analytics():
    items = log_table.scan()['Items']
    total = len(items)
    repeats = [i for i in items if i['result'] == 0]
    num_repeats = len(repeats)
    percent_repeats = num_repeats / total if total else 0
    avg_sim = sum(i['similarity_score'] for i in repeats if i.get('similarity_score')) / num_repeats if num_repeats else 0
    lengths = [len(decrypt_data(i['encrypted_prompt'])) for i in items]
    timestamps = [i['timestamp'] for i in items]
    return {
        'total_checks': total,
        'num_repeats': num_repeats,
        'percent_repeats': percent_repeats,
        'avg_similarity': avg_sim,
        'length_distribution': lengths,
        'timestamps': timestamps
    }
``` 