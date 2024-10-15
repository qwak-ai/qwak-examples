import os
import sys
import requests

def main():
    # Check if account name and model ID are provided
    if len(sys.argv) < 3:
        print("Usage: python test_model.py <account_name> <model_id>")
        sys.exit(1)

    # Variables
    account_name = sys.argv[1]
    model_id = sys.argv[2]
    qwak_token = os.getenv("QWAK_TOKEN")

    # Ensure QWAK_TOKEN is set
    if not qwak_token:
        print("Error: QWAK_TOKEN is not set. Please export the token or pass it as an environment variable.")
        sys.exit(1)

    # Test file
    file_path = "harvard.wav"

    # Ensure the file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    # URL for the model inference request
    url = f"https://models.{account_name}.qwak.ai/v1/{model_id}/predict"

    # Send the request
    with open(file_path, 'rb') as f:
        files = {'file': f}
        headers = {'Authorization': f'Bearer {qwak_token}'}
        response = requests.post(url, headers=headers, files=files)

    # Check the response
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print(f"Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    main()
