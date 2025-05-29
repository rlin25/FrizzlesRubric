import requests

API_URL = "http://localhost:8006/label"

print("Sir, enter prompts to test the API. Type 'exit' or 'quit' to stop.")

while True:
    prompt = input("Prompt: ")
    if prompt.strip().lower() in {"exit", "quit"}:
        print("Exiting live API tester. Goodbye, Sir.")
        break
    try:
        response = requests.post(API_URL, json={"prompt": prompt})
        if response.status_code == 200:
            print(f"Result: {response.text.strip()}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}") 