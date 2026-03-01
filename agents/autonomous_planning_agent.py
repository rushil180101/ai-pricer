import json
from agents.base import Agent
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from typing import Any, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"


class AutonomousPlanningAgent(Agent):

    name = "autonomous_planning_agent"

    # Defining tools

    scan_deals_function = {
        "name": "scan_deals",
        "description": "Scan the rss feeds for deals on various products and returns top deals",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_price_function = {
        "name": "estimate_price",
        "description": "Estimates the price of the product based on the given description",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    notify_user_function = {
        "name": "notify_user",
        "description": "Notifies a user about top product deals",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    # Define prompts

    system_prompt = """
    You are an agent responsible for finding product deals from rss feeds and notifying
    user about the best deal.
    """

    user_prompt = """
    First scan rss feeds for deals on products. After that, estimate the actual prices of products.
    Then notify user about the best deals. Use the provided tools to perform the above activities.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def __init__(self) -> None:
        self.logger = self.get_logger(self.name)
        self.scanner_agent = ScannerAgent()
        self.ensemble_agent = EnsembleAgent()
        self.messaging_agent = MessagingAgent()
        self.model = MODEL
        self.openai_client = OpenAI()
        self.deals = []
        self.deal_prices = []
        self.actual_prices = []

    def scan_deals(self) -> str:
        self.logger.info("Scanning rss feeds for deals")
        deals = self.scanner_agent.scan_deals().deals
        self.logger.info(f"Found {len(deals)} deals")
        for deal in deals:
            self.deals.append(deal.model_dump_json())
        return "Rss feeds scanned successfully for deals, we can proceed with estimating actual price of each product"

    def estimate_price(self) -> str:
        self.logger.info("Estimating prices of products")
        for deal in self.deals:
            # Extract deal price
            deal_data = json.loads(deal)
            deal_price = deal_data["price"]
            self.deal_prices.append(deal_price)

            # Remove deal price info from deal info to predict actual price
            deal_data.pop("price", None)
            product_description = "\n".join(
                [
                    f"Title: {deal_data['title']}",
                    f"Summary: {deal_data['summary']}",
                ]
            )
            actual_price = self.ensemble_agent.price(product_description)
            self.actual_prices.append(actual_price)

        self.logger.info(f"Estimated prices of products")
        return "Estimated prices of each product, we can proceed with notifying user about the best deals"

    def notify_user(self) -> str:
        self.logger.info("Notifying user about the best deals")
        data = zip(self.deal_prices, self.actual_prices, self.deals)
        data = sorted(data, key=lambda x: x[1] - x[0], reverse=True)
        response = None
        for i in range(min(2, len(data))):
            product_details = data[i][2]
            response = self.messaging_agent.notify(product_details)
        self.logger.info("Notifications sent successfully")
        return response

    def get_tools(self) -> List[dict]:
        return [
            {"type": "function", "function": self.scan_deals_function},
            {"type": "function", "function": self.estimate_price_function},
            {"type": "function", "function": self.notify_user_function},
        ]

    def handle_tool_call(self, message: Any) -> List[dict]:
        tool_name_to_function_mapper = {
            "scan_deals": self.scan_deals,
            "estimate_price": self.estimate_price,
            "notify_user": self.notify_user,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_func = tool_name_to_function_mapper.get(tool_name)
            tool_resp = tool_func(**tool_args)
            results.append(
                {"role": "tool", "content": tool_resp, "tool_call_id": tool_call.id}
            )
        return results

    def execute(self) -> None:
        self.logger.info("Starting execution")
        done = False
        messages = self.messages
        while not done:
            response = self.openai_client.chat.completions.create(
                messages=messages,
                model=self.model,
                tools=self.get_tools(),
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                self.logger.info(f"Calling tools")
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                self.logger.info("No tool calling")
                done = True
        reply = response.choices[0].message.content
        self.logger.info("Agent execution completed")
        return reply
