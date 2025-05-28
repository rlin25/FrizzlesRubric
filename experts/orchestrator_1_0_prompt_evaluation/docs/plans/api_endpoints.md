# API Endpoint Design

- POST /orchestrate
  - Accepts JSON body: { "prompt": <string> }
    - prompt: string, required, the user prompt to be evaluated by all experts
  - Returns JSON object with expert results
    - Each key is the expert name, value is a binary integer (0 or 1)
    - Example: { "expert_1_clarity": 1, "expert_2_documentation": 0, ... } 