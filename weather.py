import requests
import json
from datetime import datetime

# Function to fetch weather data from OpenWeatherMap API
def fetch_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        # Send GET request to the API
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse JSON response
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"Error fetching data: {err}")
        return None

# Function to display weather data in a user-friendly format
def display_weather(data):
    if not data:
        print("No data to display.")
        return
    
    # Extract relevant information
    city = data.get("name")
    country = data.get("sys", {}).get("country")
    temp = data.get("main", {}).get("temp")
    feels_like = data.get("main", {}).get("feels_like")
    humidity = data.get("main", {}).get("humidity")
    weather_desc = data.get("weather", [{}])[0].get("description", "").capitalize()
    wind_speed = data.get("wind", {}).get("speed")
    timestamp = data.get("dt")
    
    # Convert timestamp to readable date and time
    date_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    # Display formatted output
    print("\n" + "="*50)
    print(f"Weather in {city}, {country}")
    print("="*50)
    print(f"Date and Time: {date_time}")
    print(f"Temperature: {temp}°C")
    print(f"Feels Like: {feels_like}°C")
    print(f"Weather: {weather_desc}")
    print(f"Humidity: {humidity}%")
    print(f"Wind Speed: {wind_speed} m/s")
    print("="*50 + "\n")

# Main function
def main():
    # Replace with your OpenWeatherMap API key
    api_key = "YOUR_API_KEY"  # Get your free API key from https://openweathermap.org/
    city = input("Enter city name (e.g., London): ").strip()
    
    if not api_key or api_key == "YOUR_API_KEY":
        print("Please provide a valid OpenWeatherMap API key.")
        return
    
    # Fetch and display weather data
    weather_data = fetch_weather(city, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()