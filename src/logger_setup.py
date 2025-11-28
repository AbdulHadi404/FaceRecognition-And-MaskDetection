"""
Logging Setup Module
====================

Provides centralized logging configuration for all modules.
Logs are written to files for debugging executable issues.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logger(module_name, log_level=logging.DEBUG):
    """
    Set up a logger for a module that writes to both file and console.
    
    Args:
        module_name: Name of the module (e.g., 'face_recognition', 'collect_images')
        log_level: Logging level (default: DEBUG)
        
    Returns:
        Configured logger instance
    """
    # Determine log directory (works for both dev and PyInstaller)
    try:
        if getattr(sys, 'frozen', False):
            # Running as executable - log to executable directory
            exe_dir = os.path.dirname(sys.executable)
            log_dir = os.path.join(exe_dir, 'logs')
        else:
            # Running in development mode - log to project root
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_path, 'logs')
    except Exception as e:
        # Fallback to user directory
        log_dir = os.path.join(os.path.expanduser('~'), 'FaceRecognitionApp', 'logs')
    
    # Create logs directory if it doesn't exist
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        # Fallback to temp directory
        import tempfile
        log_dir = os.path.join(tempfile.gettempdir(), 'FaceRecognitionApp', 'logs')
        os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - detailed log with date in filename
    log_filename = f"{module_name}_{datetime.now().strftime('%Y%m%d')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    actual_log_path = None
    try:
        file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        actual_log_path = log_filepath
    except Exception as e:
        # If we can't write to file, try temp directory
        try:
            import tempfile
            temp_log_dir = os.path.join(tempfile.gettempdir(), 'FaceRecognitionApp_logs')
            os.makedirs(temp_log_dir, exist_ok=True)
            temp_log_path = os.path.join(temp_log_dir, log_filename)
            file_handler = logging.FileHandler(temp_log_path, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            actual_log_path = temp_log_path
        except Exception as e2:
            # If we can't log to file at all, continue without file logging
            actual_log_path = None
    
    # Console handler - simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Log initialization (use print for first message in case logging fails)
    if actual_log_path:
        print(f"[{module_name}] Log file: {actual_log_path}")
    else:
        print(f"[{module_name}] WARNING: Could not create log file - logging to console only")
    
    logger.info(f"Logger initialized for {module_name}")
    if actual_log_path:
        logger.debug(f"Log file location: {actual_log_path}")
    else:
        logger.warning("Log file creation failed - logging to console only")
    logger.debug(f"Running as executable: {getattr(sys, 'frozen', False)}")
    if hasattr(sys, '_MEIPASS'):
        logger.debug(f"PyInstaller temp path: {sys._MEIPASS}")
    logger.debug(f"Executable path: {sys.executable if getattr(sys, 'frozen', False) else 'N/A (dev mode)'}")
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"Python version: {sys.version}")
    
    return logger

