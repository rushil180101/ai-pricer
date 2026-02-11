import json
import random
import time
from typing import List
from dataset.custom_dataset_downloader import download_custom_dataset
from common.constants import (
    HF_PREPROCESSED_DATASET_REPO_ID,
    OPENAI_API_KEY,
)
from common.loggers import fine_tune_frontier_logger
from models.item import Item
from fine_tune_frontier.constants import (
    FINE_TUNING_TRAIN_MAX_DATAPOINTS,
    FINE_TUNING_VAL_MAX_DATAPOINTS,
    FINE_TUNING_MODEL,
    USER_PROMPT,
)
from openai import OpenAI


class FineTuneHandler:

    def __init__(
        self,
        train_ds: List[Item],
        val_ds: List[Item],
        test_ds: List[Item],
        train_file_path: str,
        val_file_path: str,
        test_file_path: str,
        train_max_datapoints: int = FINE_TUNING_TRAIN_MAX_DATAPOINTS,
        val_max_datapoints: int = FINE_TUNING_VAL_MAX_DATAPOINTS,
    ) -> None:
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path
        self.train_max_datapoints = train_max_datapoints
        self.val_max_datapoints = val_max_datapoints
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.train_file_id = None
        self.val_file_id = None
        self.fine_tuned_model = None

    def make_jsonl(self, item: Item) -> str:
        messages = [
            {"role": "user", "content": USER_PROMPT.format(summary=item.summary)},
            {"role": "assistant", "content": f"${item.price:.2f}"},
        ]
        jsonl = {"messages": messages}
        jsonl = json.dumps(jsonl)
        return jsonl

    def write_datapoints(self, items: List[Item], file_path: str) -> None:
        fine_tune_frontier_logger.info(
            f"Proceeding to write {len(items)} items for fine tune training"
        )
        with open(file_path, "w") as f:
            for item in items:
                jsonl = self.make_jsonl(item)
                f.write(f"{jsonl}\n")
        fine_tune_frontier_logger.info(
            f"Written {len(items)} items for fine tune training"
        )

    def write_dataset(self) -> None:
        # Write train dataset
        train_items = self.train_ds[: self.train_max_datapoints]
        train_file_path = self.train_file_path
        self.write_datapoints(train_items, train_file_path)

        # Write val dataset
        val_items = self.val_ds[: self.val_max_datapoints]
        val_file_path = self.val_file_path
        self.write_datapoints(val_items, val_file_path)

    def upload_train_file(self) -> str:
        with open(self.train_file_path, "rb") as f:
            response = self.openai_client.files.create(file=f, purpose="fine-tune")
        fine_tune_frontier_logger.info("Uploaded train file")
        return response.id

    def upload_val_file(self) -> str:
        with open(self.val_file_path, "rb") as f:
            response = self.openai_client.files.create(file=f, purpose="fine-tune")
        fine_tune_frontier_logger.info("Uploaded val file")
        return response.id

    def upload_files(self) -> None:
        self.train_file_id = self.upload_train_file()
        self.val_file_id = self.upload_val_file()

    def create_fine_tuning_job(self) -> None:
        fine_tune_frontier_logger.info("Proceeding to create fine tuning job")
        response = self.openai_client.fine_tuning.jobs.create(
            model=FINE_TUNING_MODEL,
            training_file=self.train_file_id,
            validation_file=self.val_file_id,
            seed=42,
            hyperparameters={"n_epochs": 1, "batch_size": self.train_max_datapoints},
        )
        self.fine_tuning_job_id = response.id
        fine_tune_frontier_logger.info("Created fine tuning job")

    def retrieve_fine_tuning_job(self) -> None:
        fine_tune_frontier_logger.info("Proceeding to retrieve fine tuning job")
        response = self.openai_client.fine_tuning.jobs.retrieve(self.fine_tuning_job_id)
        while response.status != "succeeded":
            if response.status == "failed":
                fine_tune_frontier_logger.error(f"Fine tuning failed: {response}")
                raise Exception("Fine tuning failed")

            wait_time = 120
            fine_tune_frontier_logger.info(
                f"Retrieving status (every {wait_time} seconds) ..."
            )
            time.sleep(wait_time)
            response = self.openai_client.fine_tuning.jobs.retrieve(
                self.fine_tuning_job_id
            )

        self.fine_tuned_model = response.fine_tuned_model
        fine_tune_frontier_logger.info(f"Fine tuning completed and retrieved job")

    def fine_tune(self) -> None:
        self.write_dataset()
        self.upload_files()
        self.create_fine_tuning_job()
        self.retrieve_fine_tuning_job()
        msg = f"Fine tuned model: {self.fine_tuned_model}"
        fine_tune_frontier_logger.info(msg)
        print(msg)


if __name__ == "__main__":
    dataset = download_custom_dataset(HF_PREPROCESSED_DATASET_REPO_ID)
    train_ds, val_ds, test_ds = dataset["train"], dataset["validation"], dataset["test"]
    train_ds = [Item(**datapoint) for datapoint in train_ds]
    val_ds = [Item(**datapoint) for datapoint in val_ds]
    test_ds = [Item(**datapoint) for datapoint in test_ds]

    random.shuffle(train_ds)
    random.shuffle(val_ds)
    random.shuffle(test_ds)

    fine_tune_handler = FineTuneHandler(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_file_path="fine_tune_frontier/jsonl/fine_tune_train.jsonl",
        val_file_path="fine_tune_frontier/jsonl/fine_tune_val.jsonl",
        test_file_path="fine_tune_frontier/jsonl/fine_tune_test.jsonl",
    )
    fine_tune_handler.fine_tune()
