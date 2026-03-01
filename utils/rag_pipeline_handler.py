import json
import os
import re
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from common.loggers import get_rotating_logger
from typing import Any, List
from dataset.custom_dataset_downloader import download_custom_dataset
from dotenv import load_dotenv

load_dotenv(override=True)


class VectorDbManager:

    def __init__(
        self,
        vector_db_path: str,
        vector_db_collection_name: str,
        embedding_model_name: str,
    ) -> None:
        self.vector_db_path = vector_db_path
        self.vector_db_collection_name = vector_db_collection_name
        self.logger = get_rotating_logger("vector_db_manager", "vector_db_manager.log")
        self.embedding_model_name = embedding_model_name
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.chroma_client = None
        self.collection = None

        self.setup_vector_store()
        assert self.chroma_client is not None
        assert self.collection is not None

    @property
    def is_data_ingested(self) -> bool:
        return self.collection.count() > 0

    def vector_store_exists(self) -> bool:
        db_exists = os.path.isdir(self.vector_db_path)
        return db_exists and self.vector_db_collection_name in [
            c.name
            for c in PersistentClient(path=self.vector_db_path).list_collections()
        ]

    def create_vector_store(self) -> Any:
        self.logger.info("Creating new vector store")
        self.chroma_client = PersistentClient(path=self.vector_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            self.vector_db_collection_name
        )
        self.logger.info("Created new vector store")

    def setup_vector_store(self) -> None:
        vector_store_exists = self.vector_store_exists()
        if not vector_store_exists:
            self.create_vector_store()
        else:
            self.logger.info("Vector store already exists")
            self.chroma_client = PersistentClient(path=self.vector_db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                self.vector_db_collection_name
            )

    def embed(self, texts: List[str]) -> Any:
        embeddings = self.embedder.encode(texts).tolist()
        return embeddings

    def ingest(self, records: List[str]) -> None:
        ids, texts = [], []
        for record_id, record in enumerate(records):
            ids.append(str(record_id))
            texts.append(json.dumps(record))

        embeddings = self.embed(texts)
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts)
        self.logger.info(f"Ingested {len(ids)} records in vector store")

    def get_relevant_records(self, query: str) -> List[str]:
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents"],
        )
        relevant_records = results["documents"][0]
        return relevant_records


class RagPipelineHandler:

    system_prompt = """
        You are a product pricer who predicts the price of
        product to nearest dollar based on its description.
        Respond only with the predicted price number without
        dollar symbol. Do not include additional information.
        """

    def __init__(
        self,
        raw_dataset_name: str,
        vector_db_manager: VectorDbManager,
        chat_model_name: str,
    ) -> None:
        self.raw_dataset_name = raw_dataset_name
        self.vector_db_manager = vector_db_manager
        self.chat_model_name = chat_model_name
        self.openai_client = OpenAI()
        self.logger = get_rotating_logger(
            "rag_pipeline_handler", "rag_pipeline_handler.log"
        )
        self.setup()

    def setup(self) -> None:
        self.logger.info("Setting up rag pipeline")
        if self.vector_db_manager.is_data_ingested:
            self.logger.info("Vector db already exists with data ingested")
            return

        dataset = download_custom_dataset(self.raw_dataset_name)
        records = (
            list(dataset["train"]) + list(dataset["validation"]) + list(dataset["test"])
        )
        self.vector_db_manager.ingest(records)
        self.logger.info(
            "Finished setting up rag pipeline and ingested records into vector db"
        )

    def lookup(self, question: str) -> List[str]:
        self.logger.info("Looking up relevant records")
        relevant_records = self.vector_db_manager.get_relevant_records(question)
        self.logger.info(f"Got {len(relevant_records)} relevant records")
        return relevant_records

    def get_messages(self, question: str) -> List[dict]:
        user_prompt_content = f"Predict the price of this product\n{question}\n\n"
        user_prompt_content += "Here's some addtional context for similar products\n\n"

        records = self.lookup(question)
        for record in records:
            record = json.loads(record)
            summary = record["summary"]
            price = record["price"]
            text = f"Product summary\n{summary}\nprice is ${price}\n"
            user_prompt_content += text

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt_content},
        ]
        return messages

    def chat(self, question: str) -> str:
        messages = self.get_messages(question)
        response = self.openai_client.chat.completions.create(
            messages=messages,
            model=self.chat_model_name,
        )
        result = response.choices[0].message.content
        pattern = r"\b\d+(?:\.\d{1,2})?\b"
        matches = re.findall(pattern, result)
        if not matches:
            raise Exception(
                f"No number found in model response which can indicate the price, model response: {result}"
            )
        price = float(matches[0])
        return price
