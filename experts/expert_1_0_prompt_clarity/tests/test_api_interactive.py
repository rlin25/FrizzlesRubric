import requests

API_URL = "http://localhost:8001/predict"  # Changed to port 8001

print("Enter a prompt to test the model (type 'exit' to quit):")
while True:
    prompt = input("Prompt: ")
    if prompt.strip().lower() == 'exit':
        print("Exiting.")
        break
    if not prompt.strip():
        print("Please enter a non-empty prompt.")
        continue
    response = requests.post(API_URL, json={"prompt": prompt})
    if response.status_code == 200:
        result = response.json()
        print(f"Full API response: {result}")
        if 'predicted_class' in result:
            print(f"Predicted Class: {result['predicted_class']}")
        else:
            print(f"Unexpected response structure.")
    else:
        print(f"Error: {response.status_code} - {response.text}") 