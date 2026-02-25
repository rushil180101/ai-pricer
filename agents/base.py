import logging
from common.loggers import get_rotating_logger


class Agent:

    def get_logger(self, name: str) -> logging.Logger:
        log_file = name.replace("-", "_").strip() + ".log"
        logger = get_rotating_logger(name, log_file)
        return logger
