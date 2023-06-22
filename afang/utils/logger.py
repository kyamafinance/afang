import logging.config
import os
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger()


class ColorFormatter(logging.Formatter):
    """Logging formatter to add colors to log statements."""

    grey = "\x1b[90m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    red = "\x1b[91m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    general_format = "%(asctime)s %(levelname)-5.5s :: %(message)s"
    info_format = f"%(asctime)s {green}%(levelname)-5.5s{reset} :: %(message)s"

    FORMATS = {
        logging.DEBUG: f"{grey}{general_format}{reset}",
        logging.INFO: info_format,
        logging.WARNING: f"{yellow}{general_format}{reset}",
        logging.ERROR: f"{red}{general_format}{reset}",
        logging.CRITICAL: f"{bold_red}{general_format}{reset}",
    }

    def format(self, record):
        record.levelname = "WARN" if record.levelname == "WARNING" else record.levelname
        record.levelname = (
            "CRIT" if record.levelname == "CRITICAL" else record.levelname
        )
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(fmt=log_fmt, datefmt="%m/%d/%Y %H:%M:%S")
        return formatter.format(record)


class Logger:
    """Application wide portable logger interface."""

    @classmethod
    def get_rotating_file_handler(cls) -> RotatingFileHandler:
        """Get a rotating file handler to use in production environments.

        :return: RotatingFileHandler
        """

        rotating_file_handler = RotatingFileHandler(
            "logs/logs.log",
            maxBytes=10485760,
            backupCount=20,
            encoding="utf8",
        )
        rotating_file_handler.name = "file"
        rotating_file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s :: %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
            )
        )
        rotating_file_handler.setLevel(logging.INFO)

        return rotating_file_handler

    def setup_logger(self, default_level: int = logging.INFO) -> None:
        """Set up the application logger.

        :param default_level: the default logging level.
        :return: None
        """

        logger.setLevel(default_level)
        color_logging_handler = logging.StreamHandler()
        color_logging_handler.name = "console"
        color_logging_handler.setLevel(default_level)
        color_logging_handler.setFormatter(ColorFormatter())
        logger.addHandler(color_logging_handler)

        # add a rotating file handler if in production environment.
        if os.environ.get("ENV", "development") == "production":
            rotating_file_handler = self.get_rotating_file_handler()
            logger.addHandler(rotating_file_handler)
