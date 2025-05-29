import requests

url = "http://localhost:8000/check"
data = {"prompt": "This is a test sentence."}

try:
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error during request:", e) 