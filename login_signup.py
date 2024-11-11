import json
import os
from typing import List, Tuple, Dict, Any, Optional
# Path for the JSON file
USER_DATA_FILE = "user_data.json"

# Load user data from JSON file
def load_user_data() -> Dict[str, str]:
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

# Save user data to JSON file
def save_user_data(data: Dict[str, str]) -> None:
    with open(USER_DATA_FILE, "w") as file:
        json.dump(data, file)

# Initialize user data by loading existing data
user_data: Dict[str, str] = load_user_data()

# Modified signup function
def signup_user(username: str, email: str, password: str) -> bool:
    """Add a new user to the user_data dictionary if the username is not taken and save to file."""
    if username in user_data:
        return False  # Username already exists
    user_data[username] = password
    save_user_data(user_data)  # Save updated data
    return True

# Login function to verify user credentials
def login_user(username: str, password: str) -> bool:
    """Check if the provided username and password match any entry in user_data."""
    user_data = load_user_data()  # Reload user data in case it's been updated
    return user_data.get(username) == password