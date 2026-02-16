import json
from typing import List
from pathlib import Path
from dataset.custom_dataset_downloader import download_custom_dataset
from dataset.upload_dataset import HFDatasetUploader
from common.constants import (
    HF_PREPROCESSED_DATASET_REPO_ID,
    HF_FINE_TUNE_OPEN_SOURCE_MODEL_DATASET_REPO_ID,
)
from common.loggers import fine_tune_open_source_logger
from models.item import Item
from transformers import AutoTokenizer
from fine_tune_open_source.constants import BASE_MODEL_NAME


class DatasetHandler:

    def __init__(
        self,
        train_ds: List[Item],
        val_ds: List[Item],
        test_ds: List[Item],
        base_model_name: str,
        dataset_local_storage_path: str,
        target_dataset_repo_id: str,
    ) -> None:
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.base_model_name = base_model_name
        self.dataset_local_storage_path = dataset_local_storage_path
        self.target_dataset_repo_id = target_dataset_repo_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.records = []

    def add_prompts(self) -> None:
        self.records = []
        complete_ds = self.train_ds + self.val_ds + self.test_ds
        for item in complete_ds:
            item.make_prompt(tokenizer=self.tokenizer)
            self.records.append(item)
        fine_tune_open_source_logger.info(
            f"Added prompts and completion for {len(self.records)} items"
        )

    def save_to_local_storage(self) -> None:
        with open(Path(self.dataset_local_storage_path) / "dataset.jsonl", "w") as f:
            for item in self.records:
                item = json.dumps(item.to_dict())
                f.write(f"{item}\n")
        fine_tune_open_source_logger.info(
            f"Saved {len(self.records)} items to local storage"
        )

    def upload_prompted_dataset_to_target_repo(self) -> None:
        dataset_uploader = HFDatasetUploader(
            repo_id=self.target_dataset_repo_id,
            private=False,
        )
        dataset_uploader.upload_prompt_dataset(
            dataset_dir=self.dataset_local_storage_path,
            should_round=True,
        )
        fine_tune_open_source_logger.info("Uploaded prompted dataset to huggingface")

    def process(self) -> None:
        self.add_prompts()
        self.save_to_local_storage()
        self.upload_prompted_dataset_to_target_repo()


if __name__ == "__main__":
    dataset = download_custom_dataset(HF_PREPROCESSED_DATASET_REPO_ID)
    train_ds, val_ds, test_ds = dataset["train"], dataset["validation"], dataset["test"]
    train_ds = [Item(**datapoint) for datapoint in train_ds]
    val_ds = [Item(**datapoint) for datapoint in val_ds]
    test_ds = [Item(**datapoint) for datapoint in test_ds]

    dataset_handler = DatasetHandler(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        base_model_name=BASE_MODEL_NAME,
        dataset_local_storage_path="fine_tune_open_source/dataset_with_prompts",
        target_dataset_repo_id=HF_FINE_TUNE_OPEN_SOURCE_MODEL_DATASET_REPO_ID,
    )
    dataset_handler.process()
