import logging
from pathlib import Path

def setup_logging(settings):
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    # Set up logger for the application
    logger = logging.getLogger("trading_bot")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Create formatters and add it to the handlers
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Create file handler
    fh = logging.FileHandler(settings.LOG_FILE)
    fh.setLevel(getattr(logging, settings.LOG_LEVEL))
    fh.setFormatter(formatter)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, settings.LOG_LEVEL))
    ch.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger 