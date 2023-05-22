import logging.config
import os
import pathlib
from logging.handlers import RotatingFileHandler

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger()


class Logger:
    """Application wide portable logger interface."""

    def __init__(self) -> None:
        """Initialize Logger class."""

        self.default_config = os.path.join(
            pathlib.Path(__file__).parents[2], "config/logging_config.yaml"
        )

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
        """Set up the application logger from the configuration present in
        config/logging_config.yaml. If the configuration yaml file does not
        exist, the logger defaults to a basic config.

        :param default_level: the default logging level to be set if the
                configuration file is unavailable.
        :return: None
        """

        path = self.default_config
        if os.path.exists(path):
            with open(path) as file:
                config = yaml.safe_load(file.read())
                logging.config.dictConfig(config)
                logging.captureWarnings(True)

                # add a rotating file handler if in production environment.
                if os.environ.get("ENV", "development") == "production":
                    rotating_file_handler = self.get_rotating_file_handler()
                    logger.addHandler(rotating_file_handler)
        else:
            logging.basicConfig(level=default_level)
