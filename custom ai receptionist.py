# import requests
# import json

# # API endpoint
# url = "https://ollama.com/api/chat"

# # Headers with your API key
# headers = {
#     "Authorization": "Bearer 3ace396abed34e5f9988e6ad2dc9bffb.ZfSKv4hQd8pMGUNCAu-xZoMY",  # 🔒 Replace with your actual key
#     "Content-Type": "application/json"
# }

# # Initial conversation history
# messages = [
#     {"role": "system", "content": "You are a friendly and professional human receptionist at an agency. Assist visitors with scheduling appointments, answering general questions about services, and providing a welcoming experience."}
# ]

# # Choose model
# model = "gpt-oss:120b"

# print("👋 Hello! I’m your AI Receptionist. How may I assist you today?\n")
# print("You can ask me about our services, schedule an appointment, or get more information. Type 'exit' to end the conversation.\n")

# while True:
#     # Collect user input
#     user_input = input("You: ")

#     if user_input.lower() in ['exit', 'quit']:
#         print("👋 Thank you for visiting! Have a great day!")
#         break

#     # Append user's message to the conversation history
#     messages.append({"role": "user", "content": user_input})

#     # Payload for the request
#     payload = {
#         "model": model,
#         "messages": messages,
#         "stream": False
#     }

#     try:
#         # Send the API request
#         response = requests.post(url, headers=headers, data=json.dumps(payload))

#         if response.status_code == 200:
#             data = response.json()

#             # Get the assistant's response
#             assistant_message = data.get("message", {}).get("content", "")

#             # Append assistant's response to the conversation history
#             messages.append({"role": "assistant", "content": assistant_message})

#             # Display the assistant's reply
#             print("\n🤖 AI Receptionist:", assistant_message, "\n")
#         else:
#             print("❌ Error:", response.status_code)
#             print("Details:", response.text)

#     except Exception as e:
#         print("⚠️ Exception occurred:", str(e))


import requests
import json
import openpyxl
import os

# ========== CONFIGURATION ==========

# API endpoint
url = "https://ollama.com/api/chat"

# Headers with your API key
headers = {
    "Authorization": "Bearer 3ace396abed34e5f9988e6ad2dc9bffb.ZfSKv4hQd8pMGUNCAu-xZoMY",  # Replace with your actual key
    "Content-Type": "application/json"
}

# Choose model
model = "gpt-oss:120b"

# Initial conversation history
messages = [
    {
        "role": "system",
        "content": (
            "You are a friendly and professional human receptionist at an agency. "
            "Assist visitors with scheduling appointments, answering general questions about services, "
            "and providing a welcoming experience.\n"
            "When a user books an appointment, reply with this format:\n"
            "'Appointment booked for [Name] on [Date] at [Time] for [Service].'"
        )
    }
]

# ========== FUNCTION TO SAVE TO EXCEL ==========

def save_appointment_to_excel(name, service, date, time):
    file_name = "appointments.xlsx"

    # Create workbook with headers if not exists
    if not os.path.exists(file_name):
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.append(["Name", "Service", "Date", "Time"])
        wb.save(file_name)

    # Append new row
    wb = openpyxl.load_workbook(file_name)
    sheet = wb.active
    sheet.append([name, service, date, time])
    wb.save(file_name)
    print("✅ Appointment .\n")

# ========== MAIN CHAT LOOP ==========

print("👋 Hello! I’m your AI Receptionist. How may I assist you today?\n")
print("You can ask me about our services, schedule an appointment, or get more information. Type 'exit' to end the conversation.\n")

while True:
    # Get user input
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit']:
        print("👋 Thank you for visiting! Have a great day!")
        break

    # Add user message to conversation
    messages.append({"role": "user", "content": user_input})

    # Prepare API payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        # Send request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            data = response.json()

            # Get assistant response
            assistant_message = data.get("message", {}).get("content", "")
            messages.append({"role": "assistant", "content": assistant_message})

            # Print assistant response
            print("\n🤖 AI Receptionist:", assistant_message, "\n")

            # ====== Basic Appointment Detection and Extraction ======

            if "appointment booked" in assistant_message.lower():
                try:
                    # Expected format:
                    # "Appointment booked for John on 2025-09-20 at 10:00AM for Consultation."
                    parts = assistant_message.split("Appointment booked for ")[1]
                    name_part, rest = parts.split(" on ", 1)
                    date_part, rest = rest.split(" at ", 1)
                    time_part, service_part = rest.split(" for ", 1)

                    name = name_part.strip()
                    date = date_part.strip()
                    time = time_part.strip().rstrip(".")
                    service = service_part.strip().rstrip(".")

                    save_appointment_to_excel(name, service, date, time)

                except Exception as e:
                    print("⚠️ Could not extract appointment details. Error:", str(e))

        else:
            print("❌ Error:", response.status_code)
            print("Details:", response.text)

    except Exception as e:
        print("⚠️ Exception occurred:", str(e))

