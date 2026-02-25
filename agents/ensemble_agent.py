from agents.base import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent


class EnsembleAgent(Agent):

    name = "ensemble_agent"

    def __init__(self) -> None:
        self.logger = self.get_logger(self.name)
        self.specialist_agent = SpecialistAgent()
        self.frontier_agent = FrontierAgent()

    def price(self, description: str) -> float:
        self.logger.info(f"{self.name} called, predicting price of the product")
        price1 = self.specialist_agent.price(description)
        price2 = self.frontier_agent.price(description)
        # Combining - average
        price = (price1 + price2) / 2.0
        self.logger.info(f"{self.name} predicted price is ${price}")
        return price
