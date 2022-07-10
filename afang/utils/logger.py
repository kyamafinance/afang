import logging.config
import os
import pathlib

import yaml


class Logger:
    """Application wide portable logger interface."""

    def __init__(self) -> None:
        """Initialize Logger class."""

        self.default_config = os.path.join(
            pathlib.Path(__file__).parents[2], "config/logging_config.yaml"
        )

    def setup_logger(self, default_level: int = logging.INFO) -> None:
        """Set up the application logger from the configuration present in
        config/logging_config.yaml. If the configuration yaml file does not
        exist, the logger defaults to a basic config.

        :param default_level: the default logging level to be set if the configuration file is unavailable.

        :return: None
        """

        path = self.default_config
        if os.path.exists(path):
            with open(path) as file:
                config = yaml.safe_load(file.read())
                logging.config.dictConfig(config)
                logging.captureWarnings(True)
        else:
            logging.basicConfig(level=default_level)
