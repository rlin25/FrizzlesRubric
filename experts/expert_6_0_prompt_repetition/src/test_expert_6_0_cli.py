import requests

API_URL = 'http://localhost:8007/check_prompt'

print('Enter prompts to test expert_6_0 (Ctrl+C to exit):')
while True:
    try:
        prompt = input('Prompt: ')
        response = requests.post(API_URL, json={'prompt': prompt})
        if response.ok:
            print('Result:', response.json().get('result'))
        else:
            print('Error:', response.text)
    except KeyboardInterrupt:
        print('\nExiting.')
        break
    except Exception as e:
        print('Error:', e) 