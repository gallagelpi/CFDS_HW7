import requests
import json
import pandas as pd

# URL of the endpoint FastAPI
API_URL = "http://127.0.0.1:8000/predict/"

# Datapoint JSON 
with open("app/input.json", "r") as f:
    datapoint = json.load(f)

def get_prediction(data):
    try:
        # Send the POST at the endpoint
        response = requests.post(API_URL, json=data, timeout=10)
        response.raise_for_status()  

        # Parseamos JSON de la respuesta
        result = response.json()
        print("Prediction result:")
        print(json.dumps(result, indent=2))

    except requests.exceptions.Timeout:
        print("Error: Request timed out.")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is the server running?")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An unexpected error occurred: {e}")
    except json.JSONDecodeError:
        print("Error: Response is not valid JSON.")

if __name__ == "__main__":
    get_prediction(datapoint)
