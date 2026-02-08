from common.constants import HF_TOKEN
from huggingface_hub import login
from datasets import Dataset, load_dataset


def download_custom_dataset(hf_dataset_repo: str) -> Dataset:
    login(token=HF_TOKEN, add_to_git_credential=True)
    dataset = load_dataset(hf_dataset_repo)
    # Basic validation
    required_splits = {"train", "validation", "test"}
    missing = required_splits - set(dataset.keys())
    if missing:
        raise ValueError(
            f"Dataset is missing required splits: {missing}. "
            f"Available splits: {list(dataset.keys())}"
        )
    return dataset
