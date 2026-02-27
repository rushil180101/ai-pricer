from agents.base import Agent
from utils.deals_fetcher import DealsFetcher
from models.deals import DealSelection


class ScannerAgent(Agent):

    name = "scanner_agent"

    def __init__(self) -> None:
        self.logger = self.get_logger(self.name)
        self.deals_fetcher = DealsFetcher()

    def scan_deals(self) -> DealSelection:
        self.logger.info(f"{self.name} scanning deals")
        deal_selection = self.deals_fetcher.get_deals()
        self.logger.info(f"{self.name} got {len(deal_selection.deals)} deals")
        return deal_selection
