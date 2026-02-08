from datasets import load_dataset
from huggingface_hub import login
from common.constants import HF_TOKEN
from dataset.parser import parse
from dataset.preprocessor import DatasetPreprocessor
from typing import Any, List
from common.loggers import dataset_logger as logger

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CATEGORIES = [
    "Electronics",
    "Arts_Crafts_and_Sewing",
    "Cell_Phones_and_Accessories",
    "Gift_Cards",
    "Handmade_Products",
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "Toys_and_Games",
]
BATCH_SIZE = 250
MAX_DATAPOINTS_PER_CATEGORY = 500
SHOULD_PREPROCESS_DATA = True


class DatasetHandler:

    def __init__(self, hf_dataset_path: str, dataset_categories: List[str]) -> None:
        self.hf_dataset_path = hf_dataset_path
        self.dataset_categories = dataset_categories
        # relative path with reference to repo root
        self.dataset_storage_dir = "dataset/batch_files"
        self.preprocessed_dataset_storage_dir = "dataset/preprocesed_batch_files"
        self.batch_size = BATCH_SIZE
        self.max_datapoints_per_category = MAX_DATAPOINTS_PER_CATEGORY
        self.dataset_preprocessor = DatasetPreprocessor()

    def download_dataset_per_category(self, dataset_category: str) -> Any:
        dataset = load_dataset(
            path=self.hf_dataset_path,
            name=dataset_category,
            split="full",
            streaming=True,
        )
        dataset = dataset.remove_columns(["images", "videos"])
        return dataset

    def write_to_file(self, filename: str, data: str) -> None:
        if not data.endswith("\n"):
            data += "\n"
        with open(filename, "w") as f:
            f.write(data)

    def save_dataset_per_category(self, dataset: Any, dataset_category: str) -> None:
        count = 0
        start = 0
        data = ""
        filename_prefix = f"{self.dataset_storage_dir}/{dataset_category}"
        for datapoint in dataset:
            item = parse(datapoint=datapoint, category=dataset_category)
            if item is not None:
                item = self.dataset_preprocessor.make_jsonl(item)
                data += item + "\n"
                count += 1
                if count % self.batch_size == 0:
                    filename = f"{filename_prefix}_{start}_{count-1}.jsonl"
                    self.write_to_file(filename, data)
                    data = ""
                    start = count
                if count > self.max_datapoints_per_category:
                    break

        # Save any unwritten data (present in memory) to file
        if data and count < self.max_datapoints_per_category:
            filename = f"{filename_prefix}_{start}_{count-1}.jsonl"
            self.write_to_file(filename, data)
            data = ""

    def get_dataset(self) -> None:
        for dataset_category in self.dataset_categories:
            dataset = self.download_dataset_per_category(dataset_category)
            self.save_dataset_per_category(dataset, dataset_category)

    def preprocess(self) -> None:
        self.dataset_preprocessor.preprocess_batches(
            input_batch_files_dir=self.dataset_storage_dir,
            output_batch_files_dir=self.preprocessed_dataset_storage_dir,
        )


if __name__ == "__main__":

    login(token=HF_TOKEN, add_to_git_credential=True)
    logger.info("Start downloading dataset")
    print("Start downloading dataset")
    categories = [f"raw_meta_{c}" for c in DATASET_CATEGORIES]
    dataset_handler = DatasetHandler(
        hf_dataset_path=DATASET_NAME, dataset_categories=categories
    )
    # dataset_handler.get_dataset()
    if SHOULD_PREPROCESS_DATA:
        dataset_handler.preprocess()
    logger.info("Successfully downloaded entire dataset")
    print("Successfully downloaded entire dataset")
