import os
import requests
from common.loggers import get_rotating_logger
from dotenv import load_dotenv

load_dotenv(override=True)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingService:

    def __init__(self) -> None:
        self.logger = get_rotating_logger("messaging_service", "messaging_service.log")
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.pushover_url = PUSHOVER_URL

    def notify(self, message: str) -> None:
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": message,
        }
        response = requests.post(self.pushover_url, data=payload)
        assert response.status_code == 200
        self.logger.info("Message delivered successfully")
