import requests
import json
from bs4 import BeautifulSoup
import re

# API endpoint for AI interaction
url = "https://ollama.com/api/chat"

# Headers with your API key
headers = {
    "Authorization": "Bearer ",  # Replace with your actual API key
    "Content-Type": "application/json"
}

# Initial conversation history for the AI
messages = [
    {"role": "system", "content": "You are a friendly and professional human receptionist at an agency. Assist visitors with scheduling appointments, answering general questions about services, providing a welcoming experience, and scraping data from webpages."}
]

# Choose the AI model
model = "gpt-oss:120b"

def fetch_website_content(url):
    """
    Function to scrape the content of a webpage using BeautifulSoup.
    It tries to find an email address from the page.
    """
    try:
        # Send a request to the URL
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the title of the page
            page_title = soup.title.string if soup.title else 'No title found'
            
            # Try to find an email address using a regex pattern
            email_match = re.search(r'[\w\.-]+@[\w\.-]+', response.text)
            email = email_match.group(0) if email_match else "No email found"

            return page_title, email
        else:
            return None, "Failed to retrieve the webpage. Status code: " + str(response.status_code)
    except Exception as e:
        return None, f"Error: {str(e)}"

print("👋 Hello! I’m your AI Receptionist. How may I assist you today?\n")
print("You can ask me about our services, schedule an appointment, get more information, or request a web scrape. Type 'exit' to end the conversation.\n")

while True:
    # Collect user input
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit']:
        print("👋 Thank you for visiting! Have a great day!")
        break
    
    # Check if user wants to scrape a website
    if user_input.lower().startswith("scrape"):
        # Extract the URL after the 'scrape' command
        url_to_scrape = user_input[6:].strip()
        
        if url_to_scrape:
            # Fetch website content
            title, content = fetch_website_content(url_to_scrape)
            if title:
                print(f"\n🌐 Scraped Website - {title}:")
                print(content)
            else:
                print(f"❌ Failed to scrape the webpage. {content}")
        else:
            print("⚠️ Please provide a valid URL to scrape.")
        continue

    # Append user's message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Payload for the request to the AI model
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        # Send the API request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            data = response.json()

            # Get the assistant's response
            assistant_message = data.get("message", {}).get("content", "")

            # Append assistant's response to the conversation history
            messages.append({"role": "assistant", "content": assistant_message})

            # Display the assistant's reply
            print("\n🤖 AI Receptionist:", assistant_message, "\n")
        else:
            print("❌ Error:", response.status_code)
            print("Details:", response.text)

    except Exception as e:
        print("⚠️ Exception occurred:", str(e))
