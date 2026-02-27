from agents.base import Agent
from utils.deals_fetcher import DealsFetcher
from models.deals import Deal, DealSelection
from typing import List


class ScannerAgent(Agent):

    name = "scanner_agent"

    def __init__(self) -> None:
        self.logger = self.get_logger(self.name)
        self.deals_fetcher = DealsFetcher()

    def scan_relevant_deals(self, description: str) -> List[Deal]:
        self.logger.info(f"{self.name} getting relevant deals")
        deal_selection: DealSelection = self.deals_fetcher.get_relevant_deals(
            description
        )
        self.logger.info(f"{self.name} got {len(deal_selection.deals)} relevant deals")
        return deal_selection

    def price(self, description: str) -> float:
        self.logger.info(f"{self.name} called, predicting price of the product")
        price = 0.0
        self.logger.info(f"{self.name} predicted price is ${price}")
        return price
