import logging
import os
from datetime import datetime

# Define log file name and path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't exist
os.makedirs(log_path, exist_ok=True)

LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Configure the basic logging setup
# This will apply to the root logger and any loggers created after this
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILEPATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# Get a logger instance for this module.
# When other modules import and use 'logger', they will get a child logger
# that inherits this configuration.
logger = logging.getLogger(__name__)

# Example log to confirm logger is configured
logger.info("Logger module initialized and logging configured.")
