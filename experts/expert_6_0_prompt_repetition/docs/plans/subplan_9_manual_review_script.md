# Subplan 9: Manual Review Script

## Description
Script to review flagged prompts, adjust threshold, and export analytics.

## Steps
- List all repeat-flagged prompts with similarity scores (result == 0)
- Allow threshold adjustment and re-run of checks (recompute similarity for all prompts)
- Export analytics as CSV or JSON
- Use data from `PromptChecks` and `Prompts` tables
- Handle DynamoDB scan errors and missing data

## Output
- List of flagged prompts
- Option to adjust threshold and re-run
- Analytics export (CSV/JSON)

## Pseudocode
```python
def review_flagged_prompts(new_threshold):
    items = log_table.scan()['Items']
    flagged = [i for i in items if i['result'] == 0]
    for entry in flagged:
        prompt = decrypt_data(entry['encrypted_prompt'])
        print(f"Prompt: {prompt}, Similarity: {entry['similarity_score']}")
    # Optionally re-run similarity checks with new threshold
    # Export analytics
    with open('analytics.json', 'w') as f:
        json.dump(compute_analytics(), f)
``` 