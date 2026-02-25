import modal
from agents.base import Agent


class SpecialistAgent(Agent):

    name = "specialist_agent"

    def __init__(self) -> None:
        self.logger = self.get_logger(self.name)
        self.pricer = modal.Cls.from_name("pricer-service", "Pricer")()

    def price(self, description: str) -> float:
        self.logger.info(f"{self.name} called, predicting price of the product")
        price = self.pricer.get_price.remote(description)
        self.logger.info(f"{self.name} predicted price is ${price}")
        return price
