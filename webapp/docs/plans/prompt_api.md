# Prompt Submission and API Integration

- Prompt input form captures user prompt as string
- On submit, send POST request to orchestrator API at http://localhost:8010/orchestrate
  - Request body: { "prompt": <string> }
- Await response from orchestrator API
- Parse response JSON: { expert_1_clarity: 0|1|e, ... }
- Update visual feedback (lights) based on results
- Display errors if API call fails or response is invalid 