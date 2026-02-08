# AI Pricer
Predict the price of Amazon product by providing its description.

# Handling dataset

- To download the dataset, execute this command from repo root directory
    - This will pull the raw dataset (Amazon Reviews 2023) from huggingface and store scrubbed data in `dataset/batch_files` directory,
     in the format ready to be sent for batch processing
    - If the `SHOULD_PREPROCESS_DATA` flag is set to `True` (in dataset/data_loader.py), then it will send this data for preprocessing/rewriting in batch mode, collect the results and then store the preprocessed/rewritten data in `dataset/preprocessed_batch_files` directory

```bash
python -m dataset.data_loader
```


- To upload processed dataset to huggingface, execute this command from repo root directory
    - This will upload the dataset (combines raw data with preprocessed/rewritten data from both local dataset directories) to huggingface
    - Repo id for the huggingface dataset should be set in `.env` file (`HF_PREPROCESSED_DATASET_REPO_ID=<your-repo-id>`)
    - My dataset can be found here: https://huggingface.co/datasets/rushil180101/ai-pricer-project-preprocessed

```bash
python -m dataset.upload_dataset
```
