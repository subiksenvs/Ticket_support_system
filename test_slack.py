import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SLACK_WEBHOOK_URL")
if not url:
    print("❌ SLACK_WEBHOOK_URL not found in environment variables.")
else:
    data = {"text": ":bell: Slack integration test successful!"}
    response = requests.post(url, json=data)
    print(response.status_code, response.text)
