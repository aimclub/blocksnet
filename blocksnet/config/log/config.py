import sys
from tqdm import tqdm
from loguru import logger
from iduedu import config as iduedu_config

LOGGER_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}

tqdm.pandas()


class LogConfig:
    """LogConfig class.

    """
    def __init__(
        self,
        logger_level="INFO",
        disable_tqdm=False,
    ):
        """Initialize the instance.

        Parameters
        ----------
        logger_level : Any, default: 'INFO'
            Description.
        disable_tqdm : Any, default: False
            Description.

        Returns
        -------
        None
            Description.

        """
        self.disable_tqdm = disable_tqdm
        self.logger_level = logger_level

    def set_logger_level(self, level: str):
        """Set logger level.

        Parameters
        ----------
        level : str
            Description.

        """
        if not level in LOGGER_LEVELS:
            raise ValueError(f"Logger should be in {LOGGER_LEVELS}")
        logger.remove()
        self.logger_level = level
        logger.add(sys.stderr, level=level)

    def set_disable_tqdm(self, disable: bool):
        """Set disable tqdm.

        Parameters
        ----------
        disable : bool
            Description.

        """
        self.disable_tqdm = disable
        iduedu_config.set_enable_tqdm(not disable)


log_config = LogConfig()
