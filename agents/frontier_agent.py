from agents.base import Agent
from utils.rag_pipeline_handler import (
    VectorDbManager,
    RagPipelineHandler,
)

VECTOR_DB_PATH = "vector_db"
VECTOR_DB_COLLECTION_NAME = "products"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAW_DATASET_NAME = "rushil180101/ai-pricer-project-preprocessed"
CHAT_MODEL_NAME = "openai/gpt-oss-120b:free"


class FrontierAgent(Agent):

    name = "frontier_agent"

    def __init__(self) -> None:
        self.logger = self.get_logger(self.name)
        self.vector_db_manager = VectorDbManager(
            vector_db_path=VECTOR_DB_PATH,
            vector_db_collection_name=VECTOR_DB_COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL_NAME,
        )
        self.rag_pipeline_handler = RagPipelineHandler(
            raw_dataset_name=RAW_DATASET_NAME,
            vector_db_manager=self.vector_db_manager,
            chat_model_name=CHAT_MODEL_NAME,
        )

    def price(self, description: str) -> float:
        self.logger.info(f"{self.name} called, predicting price of the product")
        price = self.rag_pipeline_handler.chat(description)
        self.logger.info(f"{self.name} predicted price is ${price}")
        return price
