import logging
import os
import pathlib
import shutil
from logging import handlers

from afang.utils.logger import Logger

logger = logging.getLogger(__name__)

TEST_LOGS_PATH = f"{pathlib.Path(__file__).parent}/logs"


def create_logs_dir() -> None:
    try:
        os.mkdir(TEST_LOGS_PATH)
    except OSError:
        pass


def delete_logs_dir() -> None:
    try:
        shutil.rmtree(TEST_LOGS_PATH)
    except OSError:
        pass


def test_custom_logger() -> None:
    create_logs_dir()

    Logger().setup_logger()

    # test console handler
    console_handler = logging.root.manager.root.handlers[0]
    assert console_handler.name == "console"
    assert console_handler.formatter._fmt == "%(asctime)s %(levelname)s :: %(message)s"
    assert console_handler.formatter.datefmt == "%m/%d/%Y %H:%M:%S"
    assert console_handler.__class__ == logging.StreamHandler
    assert console_handler.level == getattr(logging, "INFO")

    # test file handler
    file_handler = logging.root.manager.root.handlers[1]
    assert file_handler.name == "file"
    assert file_handler.formatter._fmt == "%(asctime)s %(levelname)s :: %(message)s"
    assert file_handler.formatter.datefmt == "%m/%d/%Y %H:%M:%S"
    assert file_handler.__class__ == handlers.RotatingFileHandler
    assert file_handler.level == getattr(logging, "DEBUG")

    delete_logs_dir()
