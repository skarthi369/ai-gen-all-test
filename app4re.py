import requests
import json
import re
user=input("Enter your message: ")
# API endpoint
url = "https://ollama.com/api/chat"

# Headers
headers = {
    "Authorization": "Bearer 75abea0b9b8d4d329432a2fbb6fcf1c8.dr3IdxNsGm0IMSgpN4pKIdDt",
    "Content-Type": "application/json"
}

# Payload
payload = {
    "model": "gpt-oss:120b",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "stream": False
}

# Send POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    response_text = response.text  # Get raw JSON string
    print("Full Response Text:\n", response_text)  # Print the full response for debugging
    # Regex pattern to extract the assistant's content
    match = re.search(r'"content"\s*:\s*"([^"]+)"', response_text)
    if match:
        assistant_message = match.group(1)
        print("Assistant says:\n")
        print(assistant_message)
    else:
        print("Assistant message not found.")
else:
    print("Error:", response.status_code, response.text)


"""
C:\gptproject>python app4re.py 
Assistant says:

Hello! How can I help you today?

C:\gptproject>python app4re.py
Full Response Text:
 {"model":"gpt-oss:120b","created_at":"2025-09-15T05:08:18.109784853Z","message":{"role":"assistant","content":"Hi there! How can I help you today?","thinking":"The user just says \"Hello\". We should respond politely. Keep friendly."},"done":true,"total_duration":432850930,"prompt_eval_count":75,"eval_count":35}

Assistant says:

Hi there! How can I help you today?

C:\gptproject>"""