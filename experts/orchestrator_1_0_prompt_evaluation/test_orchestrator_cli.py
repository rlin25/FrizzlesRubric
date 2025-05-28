import requests

API_URL = 'http://localhost:8010/orchestrate'

print('Enter prompts to test the orchestrator API (Ctrl+C to exit):')
while True:
    try:
        prompt = input('Prompt: ')
        response = requests.post(API_URL, json={'prompt': prompt})
        if response.ok:
            print('Results:', response.json())
        else:
            print('Error:', response.text)
    except KeyboardInterrupt:
        print('\nExiting.')
        break
    except Exception as e:
        print('Error:', e) 