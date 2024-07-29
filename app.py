import logging
import sys

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
stdout_handler = logging.StreamHandler(sys.stdout)
stderr_handler = logging.StreamHandler(sys.stderr)

stdout_handler.setLevel(logging.DEBUG)
stderr_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)

# Example log messages
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.error('This is an error message')
