import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from sklearn.model_selection import train_test_split
from models.item import Item
from common.loggers import hf_dataset_upload_logger
from common.constants import (
    HF_TOKEN,
    HF_PREPROCESSED_DATASET_REPO_ID,
)


class HFDatasetUploader:

    def __init__(self, repo_id: str, private: bool = True) -> None:
        self.repo_id = repo_id
        self.private = private
        self.records = []
        self.items = {}

    def read_raw_dataset(self, raw_dataset_dir: str) -> None:
        hf_dataset_upload_logger.info("Proceeding to read raw dataset")
        raw_dataset_dir = Path(raw_dataset_dir)
        for file_path in raw_dataset_dir.glob("*.jsonl"):
            with open(file_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    item_id = data["custom_id"]
                    product_data = data["body"]["messages"][1]["content"]
                    product_data = json.loads(product_data)
                    item = Item(**product_data)
                    self.items[item_id] = item
        total_items = len(self.items.keys())
        hf_dataset_upload_logger.info(f"Read raw dataset with {total_items} items")

    def read_processed_dataset(self, processed_dataset_dir: str) -> None:
        hf_dataset_upload_logger.info("Proceeding to read processed dataset")
        processed_dataset_dir = Path(processed_dataset_dir)
        for file_path in processed_dataset_dir.glob("*.jsonl"):
            with open(file_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    item_id = data["custom_id"]
                    content = data["response"]["body"]["choices"][0]["message"][
                        "content"
                    ]
                    self.items[item_id].summary = content
        total_items = len(self.items.keys())
        hf_dataset_upload_logger.info(
            f"Read processed dataset with {total_items} items"
        )

    def create_dataset(self) -> None:
        hf_dataset_upload_logger.info("Proceeding to create dataset")
        self.records = []
        for _, item_obj in self.items.items():
            datapoint = item_obj.to_dict()
            self.records.append(datapoint)
        total_records = len(self.records)
        hf_dataset_upload_logger.info(f"Created dataset with {total_records} records")

    def push_dataset_to_hub(self, should_round: bool = False) -> None:
        data = self.records
        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42
        )

        if should_round:
            hf_dataset_upload_logger.info("Rounding off completion")
            for i, dp in enumerate(train_data):
                dp["completion"] = (dp["completion"].split("."))[0] + ".00"
                train_data[i] = {"prompt": dp["prompt"], "completion": dp["completion"]}
            for i, dp in enumerate(val_data):
                dp["completion"] = (dp["completion"].split("."))[0] + ".00"
                val_data[i] = {"prompt": dp["prompt"], "completion": dp["completion"]}

            test_data = [
                {"prompt": dp["prompt"], "completion": dp["completion"]}
                for dp in test_data
            ]

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
                "test": Dataset.from_list(test_data),
            }
        )
        login(token=HF_TOKEN, add_to_git_credential=True)
        dataset_dict.push_to_hub(repo_id=self.repo_id, private=self.private)
        hf_dataset_upload_logger.info("Successfully pushed dataset to huggingface")

    def upload(self, raw_dataset_dir: str, processed_dataset_dir: str) -> None:
        self.read_raw_dataset(raw_dataset_dir=raw_dataset_dir)
        self.read_processed_dataset(processed_dataset_dir=processed_dataset_dir)
        self.create_dataset()
        self.push_dataset_to_hub()

    def read_prompt_dataset(self, dataset_dir: str) -> None:
        hf_dataset_upload_logger.info("Proceeding to read simple dataset")
        dataset_dir = Path(dataset_dir)
        for file_path in dataset_dir.glob("*.jsonl"):
            with open(file_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    item_id = data["item_id"]
                    item = Item(**data)
                    self.items[item_id] = item
        total_items = len(self.items.keys())
        hf_dataset_upload_logger.info(f"Read simple dataset with {total_items} items")

    def upload_prompt_dataset(
        self, dataset_dir: str, should_round: bool = False
    ) -> None:
        self.read_prompt_dataset(dataset_dir=dataset_dir)
        self.create_dataset()
        self.push_dataset_to_hub(should_round=should_round)


if __name__ == "__main__":
    hf_dataset_uploader = HFDatasetUploader(
        repo_id=HF_PREPROCESSED_DATASET_REPO_ID,
        private=False,
    )
    # relative path wrt repo root
    hf_dataset_uploader.upload(
        raw_dataset_dir="dataset/batch_files",
        processed_dataset_dir="dataset/preprocessed_batch_files",
    )
