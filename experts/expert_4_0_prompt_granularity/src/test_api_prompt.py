import requests

API_URL = "http://localhost:8005/predict"

if __name__ == "__main__":
    print("Enter prompts to test the API (type 'exit' to quit):")
    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == "exit":
            break
        try:
            response = requests.post(API_URL, json={"prompt": prompt})
            print(f"API Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}") 