import requests
import sys

API_URL = "http://localhost:8002/predict"

def main():
    print("Enter a prompt to classify (type 'exit' to quit):", flush=True)
    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() == 'exit':
                print("Exiting.")
                break
            if not prompt:
                continue
            response = requests.post(API_URL, json={"prompt": prompt})
            if response.status_code == 200:
                result = response.json().get("result")
                print(f"Result: {result}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    main() 