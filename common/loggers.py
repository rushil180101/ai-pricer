import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_rotating_logger(
    name: str,
    log_file: str | Path,
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


# Loggers for different modules

dataset_logger = get_rotating_logger(
    name="dataset_logger",
    log_file="dataset_handling_logs.log",
)

hf_dataset_upload_logger = get_rotating_logger(
    name="hf_dataset_upload_logger",
    log_file="hf_dataset_uploading_logs.log",
)

fine_tune_frontier_logger = get_rotating_logger(
    name="fine_tune_frontier_logger",
    log_file="fine_tune_frontier_logs.log",
)

fine_tune_open_source_logger = get_rotating_logger(
    name="fine_tune_open_source_logger",
    log_file="fine_tune_open_source_logs.log",
)
