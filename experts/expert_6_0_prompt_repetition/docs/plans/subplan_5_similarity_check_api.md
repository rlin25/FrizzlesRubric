# Subplan 5: Similarity Check API

## Description
API to check if a prompt is excessively similar to any stored prompt.

## Steps
- Input: raw prompt (string)
- Enforce length limit (Subplan 6)
- Preprocess prompt (Subplan 1)
- Generate embedding (Subplan 2)
- Decrypt all stored embeddings from DynamoDB (Subplan 3, 4)
- Compute cosine similarity between new embedding and all stored embeddings
- If any similarity > fixed threshold (e.g., 0.85):
  - Log check as repeat (Subplan 7)
  - Return 0
- Else:
  - Store new prompt and embedding (Subplan 4)
  - Log check as unique (Subplan 7)
  - Return 1
- Handle API errors and invalid input

## Input
- Raw prompt (string)

## Output
- { "result": 0 or 1 }

## Pseudocode
```python
def check_prompt_api(raw_prompt):
    if len(raw_prompt) > 512:
        return {"error": "Prompt too long"}
    preprocessed = preprocess_prompt(raw_prompt)
    embedding = get_embedding(preprocessed)
    stored = get_all_embeddings()
    for item in stored:
        emb = decrypt_data(item['encrypted_embedding'])
        sim = cosine_similarity(embedding, json.loads(emb))
        if sim > 0.85:
            log_check(raw_prompt, 0, sim, item['prompt_id'])
            return {"result": 0}
    # Store new prompt
    encrypted_prompt = encrypt_data(preprocessed)
    encrypted_embedding = encrypt_data(embedding)
    store_prompt(preprocessed, encrypted_prompt, encrypted_embedding)
    log_check(raw_prompt, 1, None, None)
    return {"result": 1} 