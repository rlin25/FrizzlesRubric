import requests

ORCHESTRATOR_URL = "http://localhost:8008/flow_predict"

def main():
    print("Enter a prompt to test the orchestrator (type 'exit' to quit):")
    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == 'exit':
            print("Exiting.")
            break
        try:
            response = requests.post(ORCHESTRATOR_URL, json={"prompt": prompt}, timeout=10)
            print("Full API response:", response.json())
        except Exception as e:
            print(f"Error contacting orchestrator: {e}")

if __name__ == "__main__":
    main() 