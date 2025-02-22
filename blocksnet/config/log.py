import os
import sys
from loguru import logger
from iduedu import config as iduedu_config

LOGGER_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}


class LogConfig:
    def __init__(
        self,
        logger_level="INFO",
        disable_tqdm=False,
    ):
        self.disable_tqdm = disable_tqdm
        self.logger_level = logger_level

    def set_logger_level(self, level: str):
        if not level in LOGGER_LEVELS:
            raise ValueError(f"Logger should be in {LOGGER_LEVELS}")
        logger.remove()
        self.logger_level = level
        logger.add(sys.stderr, level=level)

    def set_disable_tqdm(self, disable: bool):
        self.disable_tqdm = disable
        iduedu_config.set_enable_tqdm(not disable)


log_config = LogConfig()
