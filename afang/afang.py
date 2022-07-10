import logging

from afang.utils.logger import Logger

# Initialize logger
Logger().setup_logger()

logger = logging.getLogger(__name__)
