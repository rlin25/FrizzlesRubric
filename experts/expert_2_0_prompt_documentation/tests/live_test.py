import requests

API_URL = "http://localhost:8003/check"

print("Live Documentation Classifier Test Console")
print("Type your prompt and press Enter. Type 'exit' to quit.\n")

while True:
    prompt = input("Enter prompt: ")
    if prompt.strip().lower() == 'exit':
        print("Exiting.")
        break
    data = {"prompt": prompt}
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            if result['prediction'] == 1:
                print(f"Prediction: Well documented")
            else:
                print(f"Prediction: Poorly documented")
            print(f"Probability: {result['probability']:.4f}\n")
        else:
            print(f"Error: Received status code {response.status_code}")
    except Exception as e:
        print(f"Request failed: {e}\n") 