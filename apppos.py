import requests

# Base URL of the FastAPI server
BASE_URL = "http://127.0.0.1:8000"

# Data to send in POST request
item_data = {
    "name": "Laptop",
    "price": 200.5
}

# Item ID to use
item_id = 2

# POST request to create an item
post_response = requests.post(f"{BASE_URL}/items/?item_id={item_id}", json=item_data)
print("POST Response:")
print(post_response.json())

# GET request to retrieve the item
get_response = requests.get(f"{BASE_URL}/items/{item_id}")
print("\nGET Response:")
print(get_response.json())