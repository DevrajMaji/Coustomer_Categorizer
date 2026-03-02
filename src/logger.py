import logging
import os
from datetime import datetime

# Constants
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create logs directory
LOGS_PATH = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(LOGS_PATH, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

