# Subplan 10: API Endpoints

## Description
API endpoints for prompt checking, analytics, and review.

## Endpoints
- POST /check_prompt
  - Input: { "prompt": string }
  - Output: { "result": 0 or 1 }
  - Error: { "error": string }
  - Pseudocode:
    ```python
    @app.route('/check_prompt', methods=['POST'])
    def check_prompt():
        data = request.get_json()
        return check_prompt_api(data['prompt'])
    ```
- GET /analytics
  - Output: analytics summary (JSON)
  - Pseudocode:
    ```python
    @app.route('/analytics', methods=['GET'])
    def analytics():
        return compute_analytics()
    ```
- GET /review
  - Output: list of flagged prompts, similarity scores, and manual threshold adjustment option
  - Pseudocode:
    ```python
    @app.route('/review', methods=['GET'])
    def review():
        return review_flagged_prompts(new_threshold=0.85)
    ``` 