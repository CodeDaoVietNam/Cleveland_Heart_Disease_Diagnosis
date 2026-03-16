import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name="app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers are already configured to avoid duplicate logs
    if not logger.handlers:
        # Console handler for terminal display
        c_handler = logging.StreamHandler()
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        
        # File handler for tracking in production
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        f_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=5000000, backupCount=5)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        f_handler.setFormatter(f_format)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger
