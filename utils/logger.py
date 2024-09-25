import logging
import os
from logging.handlers import RotatingFileHandler

class DuplicateFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_log = None

    def filter(self, record):
        current_log = (record.levelno, record.msg)
        if current_log != self.last_log:
            self.last_log = current_log
            return True
        return False

def setup_logger():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('app_logger')
    logger.setLevel(logging.INFO)

    # Correcting the logging format (removing milliseconds)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'), 
        maxBytes=2000000, 
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add duplicate message filter
    duplicate_filter = DuplicateFilter()
    logger.addFilter(duplicate_filter)

    return logger

logger = setup_logger()

if __name__ == "__main__":
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
