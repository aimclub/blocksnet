import sys
from tqdm import tqdm
from loguru import logger
from iduedu import config as iduedu_config

LOGGER_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}

tqdm.pandas()


class LogConfig:
    """Manage logging behaviour for the BlocksNet package.

    Parameters
    ----------
    logger_level : str, default="INFO"
        Loguru logging level to use for stderr output. Must be one of the
        values defined in :data:`LOGGER_LEVELS`.
    disable_tqdm : bool, default=False
        If ``True``, disables tqdm progress bars used across the library.
    """

    def __init__(
        self,
        logger_level="INFO",
        disable_tqdm=False,
    ):
        self.disable_tqdm = disable_tqdm
        self.logger_level = logger_level

    def set_logger_level(self, level: str):
        """Configure the global loguru logger level.

        Parameters
        ----------
        level : str
            Desired logging level. Must be present in :data:`LOGGER_LEVELS`.

        Raises
        ------
        ValueError
            If *level* is not recognised as a valid logging level.
        """

        if not level in LOGGER_LEVELS:
            raise ValueError(f"Logger should be in {LOGGER_LEVELS}")
        logger.remove()
        self.logger_level = level
        logger.add(sys.stderr, level=level)

    def set_disable_tqdm(self, disable: bool):
        """Toggle tqdm progress bars used during computations.

        Parameters
        ----------
        disable : bool
            Whether to disable tqdm progress bars (``True``) or keep them
            enabled (``False``).
        """

        self.disable_tqdm = disable
        iduedu_config.set_enable_tqdm(not disable)


log_config = LogConfig()
