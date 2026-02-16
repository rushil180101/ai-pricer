import random
from dataset.custom_dataset_downloader import download_custom_dataset
from common.constants import HF_FINE_TUNE_OPEN_SOURCE_MODEL_DATASET_REPO_ID
from fine_tune_open_source.constants import BASE_MODEL_NAME
from models.item import Item

if __name__ == "__main__":
    dataset = download_custom_dataset(HF_FINE_TUNE_OPEN_SOURCE_MODEL_DATASET_REPO_ID)
    train_ds, val_ds, test_ds = dataset["train"], dataset["validation"], dataset["test"]
    train_ds = [Item(**datapoint) for datapoint in train_ds]
    val_ds = [Item(**datapoint) for datapoint in val_ds]
    test_ds = [Item(**datapoint) for datapoint in test_ds]

    random.shuffle(train_ds)
    random.shuffle(val_ds)
    random.shuffle(test_ds)

    print(f"Train = {len(train_ds)}")
    print(f"Val = {len(val_ds)}")
    print(f"Test = {len(test_ds)}")
