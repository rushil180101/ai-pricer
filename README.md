---
title: AI Pricer + Deals finder
emoji: 📚
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
---

# AI Pricer + Deals Finder
- Live on huggingface: https://huggingface.co/spaces/rushil180101/ai-pricer
- Let AI find the best deals on various products
- Lets agents do the heavy lifting

# Project setup

- Install required dependencies

```bash
pip install -r requirements.txt
```

- Create `.env` file in repo root directory and set the following values

```bash

# openrouter
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=<api-key>

# huggingface
HF_TOKEN=<hf-token>
HF_PREPROCESSED_DATASET_REPO_ID=<dataset-repo-id>

# openai
OPENAI_API_KEY=<api-key>

# wandb
WANDB_API_KEY=<api-key>

# modal
MODAL_TOKEN_ID=<token-id>
MODAL_TOKEN_SECRET=<token-secret>

# pushover
PUSHOVER_USER=<user-token>
PUSHOVER_TOKEN=<app-token>

```


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
        - **Train - 3081**
        - **Validation - 385**
        - **Test - 386**
    - Dataset with `prompt` and `completion`: rushil180101/ai-pricer-fine-tune-open-source-model

```bash
python -m dataset.upload_dataset
```

# Models performance comparison

- The performance of various models can be checked within the `arena` directory.

![Bar Chart](arena/model_comparison_results.png)

# Agentic project

- Multiple agents contribute in price prediction.
    - Autonomous planning agent - Task orchestration
    - Scanner agent - Scans rss feeds for deals
    - Ensemble agent - Combination of fine-tuned model and RAG
        - Frontier agent - Uses RAG + frontier models for price prediction
        - Specialist agent - Uses fine-tuned model for price prediction (This is deployed on https://modal.com/)
    - Messaging agent - Notifies user for best product deals (https://pushover.net/)
- Agents' code can be viewed within `agents` directory.

- Datasets
    - Original Amazon products dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
    - Custom preprocessed/rewritten dataset: https://huggingface.co/datasets/rushil180101/ai-pricer-project-preprocessed
    - Custom dataset used for fine-tuning: https://huggingface.co/datasets/rushil180101/ai-pricer-fine-tune-open-source-model

- Models
    - Fine-tuned model: https://huggingface.co/rushil180101/pricer-2026-02-19T15-32-44

![Sample output](project_output.png)
