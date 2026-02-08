import concurrent.futures
import json
import time
from models.item import Item
from common.constants import (
    OPENAI_API_KEY,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_MODEL,
)
from common.loggers import dataset_logger as logger
from typing import Tuple, List
from openai import OpenAI
from pathlib import Path

SHOULD_USE_LOCAL_OLLAMA_MODEL = False
PREPROCESSING_MODEL = "gpt-4.1-nano"
BATCH_LIMIT = 2
TEXT_PREPROCESSING_SYSTEM_PROMPT = """
Create a concise description of a product based on the provided details. Respond in the following format
Title: title of the product (for example, Microwave oven)
Category: category of the product (for example, Electronics)
Brand: brand name
Description: short description of the product (1 sentence)
Details: product features in one line
"""


class DatasetPreprocessor:

    def __init__(self) -> None:
        if SHOULD_USE_LOCAL_OLLAMA_MODEL:
            self.openai_client = OpenAI(base_url=LOCAL_OLLAMA_BASE_URL)
            self.model = LOCAL_OLLAMA_MODEL
        else:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = PREPROCESSING_MODEL

        self.batch_limit = BATCH_LIMIT
        self.file_ids = {}
        self.batch_ids = []
        self.results = []

    def make_jsonl(self, item: Item) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TEXT_PREPROCESSING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Product data: {item.to_json()}"},
            ],
            "max_tokens": 1000,
        }
        line = {
            "custom_id": str(item.item_id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    def retrieve_batch(self, batch_id: str) -> Tuple[str, str]:
        try:
            result = self.openai_client.batches.retrieve(batch_id=batch_id)
            while result.status != "completed":
                if result.status == "failed":
                    logger.info(f"Errors = {result.errors}")
                    return None, None

                time.sleep(5)
                result = self.openai_client.batches.retrieve(batch_id=batch_id)
            batch_input_file_id = result.input_file_id
            batch_output_file_id = result.output_file_id
            return (batch_input_file_id, batch_output_file_id)
        except Exception as exc:
            logger.info(f"Failed to process batch {batch_id}, error: {exc}")
            return None, None

    def submit_files(self, file_paths: List[Path]) -> None:
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                response = self.openai_client.files.create(file=f, purpose="batch")
                file_id = response.id
                self.file_ids[file_id] = file_path.name
        total_batch_files = len(list(self.file_ids.keys()))
        logger.info(f"Uploaded {total_batch_files} batch files for preprocessing")

    def create_batches(self) -> None:
        for file_id in self.file_ids.keys():
            response = self.openai_client.batches.create(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=file_id,
            )
            batch_id = response.id
            self.batch_ids.append(batch_id)

        logger.info(f"Created {len(self.batch_ids)} batches for preprocessing")

    def retrieve_batches(self) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as tpx:
            self.results = tpx.map(self.retrieve_batch, self.batch_ids)
        logger.info(f"Preprocessed {len(self.batch_ids)} batches and retrieved results")

    def store_batch_file_results(self, output_batch_files_dir: str) -> None:
        logger.info("Proceeding to save preprocessing results in local storage")
        logger.info(f"Results output dir: {output_batch_files_dir}")
        for input_file_id, output_file_id in self.results:
            if input_file_id is None and output_file_id is None:
                continue
            response = self.openai_client.files.content(file_id=output_file_id)
            output_file_name = self.file_ids[input_file_id]
            output_file_path = Path(output_batch_files_dir) / output_file_name
            response.write_to_file(file=output_file_path)

    def preprocess_limited_batches(
        self,
        output_batch_files_dir: str,
        file_paths: List[Path],
    ) -> bool:
        # Batch processing with 4 steps
        # 1. Create files
        # 2. Create batches for each file
        # 3. Retrieve processed batches
        # 4. Get processed file content for each batch/input file

        # Step 1: Create files
        self.submit_files(file_paths=file_paths)

        # Step 2: Create batches
        self.create_batches()

        # Step 3: Retrieve processed batches
        self.retrieve_batches()

        # Step 4: Get processed file content for each batch/input file
        self.store_batch_file_results(output_batch_files_dir=output_batch_files_dir)

    def preprocess_batches(
        self,
        input_batch_files_dir: str,
        output_batch_files_dir: str,
    ) -> None:
        # Process limited batches due to token limits on smaller models
        input_batch_files_dir = Path(input_batch_files_dir)
        logger.info(
            f"Proceeding to preprocess batch files (dir: {input_batch_files_dir})"
        )
        file_paths = []

        for file_path in input_batch_files_dir.glob("*.jsonl"):
            file_paths.append(file_path)
            if len(file_paths) >= self.batch_limit:
                self.preprocess_limited_batches(
                    output_batch_files_dir=output_batch_files_dir,
                    file_paths=file_paths,
                )
                # Reset values as new batch will be formed
                file_paths = []
                self.file_ids = {}
                self.batch_ids = []
                self.results = []
