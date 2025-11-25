# import requests
# import json
# user=input("Enter your message: ")
# # API endpoint
# url = "https://ollama.com/api/chat"

# # Headers
# headers = {
#     "Authorization": "Bearer 3ace396abed34e5f9988e6ad2dc9bffb.ZfSKv4hQd8pMGUNCAu-xZoMY",
#     "Content-Type": "application/json"
# }

# # Payload
# payload = {
#     "model": "gpt-oss:120b",
#     "messages": [
#         {"role": "user", "content": (user)"}
#     ],
#     "stream": False
# }



# # Send POST request
# response = requests.post(url, headers=headers, data=json.dumps(payload))

# # Process response
# if response.status_code == 200:
#     data = response.json()
#     # Isolate the assistant's message
#     assistant_message = data.get("message", {}).get("content", "")
    
#     # Display it clearly
#     print("Assistant says:\n")
#     print(assistant_message)
# else:
#     print("Error:", response.status_code, response.text)
import requests
import json

user = input("Enter your message: ")

# API endpoint
url = "https://ollama.com/api/chat"

# Headers
headers = {
    "Authorization": "Bearer 3ace396abed34e5f9988e6ad2dc9bffb.ZfSKv4hQd8pMGUNCAu-xZoMY",
    "Content-Type": "application/json"
}

# Payload
payload = {
    "model": "gpt-oss:120b",
    "messages": [
        {"role": "user", "content": user}
    ],
    "stream": False
}

# Send POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Process response
if response.status_code == 200:
    data = response.json()
    # Isolate the assistant's message
    assistant_message = data.get("message", {}).get("content", "")
    
    # Display it clearly
    print("Assistant says:\n")
    print(assistant_message)
else:
    print("Error:", response.status_code, response.text)