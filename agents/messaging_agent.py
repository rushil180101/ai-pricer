import os
import requests
from agents.base import Agent
from dotenv import load_dotenv
from http import HTTPStatus
from openai import OpenAI

load_dotenv(override=True)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

SYSTEM_PROMPT = "You are responsible for rewriting the product description as if notifying a user about the deal"
MODEL = "gpt-4.1-nano"


class MessagingAgent(Agent):

    name = "messaging_agent"

    def __init__(self) -> None:
        self.logger = self.get_logger(self.name)
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.pushover_url = PUSHOVER_URL
        self.openai_client = OpenAI()

    def rewrite(self, product_description: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Rewrite the product description in a human readable format: {product_description}",
            },
        ]
        response = self.openai_client.chat.completions.create(
            messages=messages,
            model=MODEL,
        )
        rewritten = response.choices[0].message.content
        return rewritten

    def notify(self, product_description: str) -> str:
        message = self.rewrite(product_description)
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": message,
        }
        response = requests.post(self.pushover_url, data=payload)
        assert response.status_code == HTTPStatus.OK
        self.logger.info("Message delivered successfully")
        return message
