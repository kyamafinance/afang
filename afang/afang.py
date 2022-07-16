import logging

from utils.logger import Logger

# Initialize logger
Logger().setup_logger()

logger = logging.getLogger(__name__)
