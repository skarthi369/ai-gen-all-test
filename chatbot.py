import requests
import json

# API endpoint
url = "https://ollama.com/api/chat"

# Headers with your API key
headers = {
    "Authorization": "Bearer 3ace396abed34e5f9988e6ad2dc9bffb.ZfSKv4hQd8pMGUNCAu-xZoMY",  # 🔒 Replace with your actual key
    "Content-Type": "application/json"
}

# Initial message history
messages = []

# Choose model
model = "gpt-oss:120b"

print("🤖 Welcome to the Chatbot! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit']:
        print("👋 Goodbye!")
        break

    # Append user message
    messages.append({"role": "user", "content": user_input})

    # Payload for the request
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        # Make API request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            data = response.json()

            # Get assistant message
            assistant_message = data.get("message", {}).get("content", "")

            # Add assistant message to history
            messages.append({"role": "assistant", "content": assistant_message})

            # Print assistant's reply
            print("\n🤖 Bot:", assistant_message, "\n")
        else:
            print("❌ Error:", response.status_code)
            print("Details:", response.text)

    except Exception as e:
        print("⚠️ Exception occurred:", str(e))
