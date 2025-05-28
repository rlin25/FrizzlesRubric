# Expert Endpoint Configuration

- expert_1_clarity: http://172.31.48.199:8008/check_prompt
- expert_2_documentation: http://172.31.48.99:8003/check_prompt
- expert_3_structure: http://172.31.48.22:8004/check_prompt
- expert_4_granulation: http://172.31.48.104:8005/check_prompt
- expert_5_tooling: http://172.31.48.12:8006/check_prompt
- expert_6_repetition: http://172.31.48.208:8007/check_prompt
- Each endpoint expects a POST request with JSON body: { "prompt": <string> }
- Each endpoint returns JSON: { "result": <0 or 1> } 