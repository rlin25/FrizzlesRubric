import requests
import json
import time

def test_grammar_check(prompt):
    url = "http://localhost:8000/check"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def test_health():
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def main():
    # Test health endpoint
    print("\nTesting health endpoint...")
    health_response = test_health()
    print(f"Health check response: {health_response}")
    
    # Test grammar check with various prompts
    test_prompts = [
        "This is a grammatically correct sentence.",
        "Me going to store now.",
        "The quick brown fox jumps over the lazy dog.",
        "They is going to the party.",
        "I have been working on this project for three months.",
        "",  # Empty string
        "   ",  # Whitespace only
        "A" * 1001,  # Exceeds max length
    ]
    
    print("\nTesting grammar check endpoint...")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        result = test_grammar_check(prompt)
        if result:
            print(f"Response: {json.dumps(result, indent=2)}")
        time.sleep(0.5)  # Rate limiting

if __name__ == "__main__":
    main() 