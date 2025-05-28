# Expert 6 - Repetition: Masterplan

---

## Subplan 1: Preprocessing
- Lowercase prompt
- Remove punctuation
- Remove extra whitespace
- Remove NLTK English stopwords
- Normalize unicode
- Output: preprocessed prompt (string)

---

## Subplan 2: Embedding Generation
- Use HuggingFace `sentence-transformers/all-MiniLM-L6-v2` model
- Input: preprocessed prompt (string)
- Output: embedding (list of floats)

---

## Subplan 3: Encryption
- Use AWS KMS to encrypt preprocessed prompt and embedding before storing in DynamoDB
- Use AWS KMS to decrypt when reading for similarity checks
- Input: preprocessed prompt (string), embedding (list of floats)
- Output: encrypted_prompt (string), encrypted_embedding (string, JSON-serialized list)

---

## Subplan 4: DynamoDB Storage
- Table: `Prompts`
  - Primary key: `prompt_id` (SHA256 hash of preprocessed prompt)
  - Attributes:
    - `prompt_id` (string)
    - `encrypted_prompt` (string, KMS-encrypted)
    - `encrypted_embedding` (string, KMS-encrypted, JSON list)
    - `created_at` (ISO timestamp)
- Store new prompt and embedding if unique
- Retrieve all stored embeddings for similarity check

---

## Subplan 5: Similarity Check API
- Input: raw prompt (string)
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

---

## Subplan 6: Length Limit
- Reject prompts exceeding a fixed character or token limit (e.g., 512 characters)
- Return error or rejection response if limit exceeded

---

## Subplan 7: Logging
- Table: `PromptChecks`
  - Attributes:
    - `check_id` (UUID)
    - `timestamp` (ISO)
    - `encrypted_prompt` (KMS-encrypted)
    - `result` (0 or 1)
    - `similarity_score` (float, if repeat)
    - `most_similar_prompt_id` (if repeat)
    - `threshold` (float)
- Log every similarity check and result

---

## Subplan 8: Analytics
- Compute and expose:
  - Total prompts checked
  - Number and percentage of repeats flagged
  - Average similarity score for repeats
  - Distribution of prompt lengths
  - Timestamps of checks
- Use data from `PromptChecks` table

---

## Subplan 9: Manual Review Script
- List all repeat-flagged prompts with similarity scores
- Allow threshold adjustment and re-run of checks
- Export analytics as CSV or JSON
- Use data from `PromptChecks` and `Prompts` tables

---

## Subplan 10: API Endpoints
- POST /check_prompt
  - Input: { "prompt": string }
  - Output: { "result": 0 or 1 }
- GET /analytics
  - Output: analytics summary (JSON)
- GET /review
  - Output: list of flagged prompts, similarity scores, and manual threshold adjustment option

---

## Subplan 11: Security
- All sensitive data encrypted with AWS KMS
- No user-identifying information stored
- Ensure proper IAM permissions for KMS and DynamoDB

---

## Subplan 12: Dependencies
- Python3
- sentence-transformers
- boto3
- nltk
- numpy
- AWS KMS permissions
- DynamoDB tables: `Prompts`, `PromptChecks` 