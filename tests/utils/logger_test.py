import logging
import os
from logging import handlers

from afang.utils.logger import Logger


def test_custom_logger(mocker) -> None:
    mocker.patch.dict(os.environ, {"ENV": "production"})
    Logger().setup_logger()

    # test console handler
    console_handler = logging.root.manager.root.handlers[0]
    assert console_handler.name == "console"
    assert console_handler.formatter._fmt == "%(message)s"
    assert console_handler.__class__ == logging.StreamHandler
    assert console_handler.level == getattr(logging, "INFO")

    # test file handler
    file_handler = logging.root.manager.root.handlers[-1]
    assert file_handler.name == "file"
    assert file_handler.formatter._fmt == "%(asctime)s %(levelname)s :: %(message)s"
    assert file_handler.formatter.datefmt == "%m/%d/%Y %H:%M:%S"
    assert file_handler.__class__ == handlers.RotatingFileHandler
    assert file_handler.level == getattr(logging, "INFO")
