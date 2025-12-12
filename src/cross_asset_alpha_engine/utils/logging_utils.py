"""Logging utilities for Cross-Asset Alpha Engine.

This module provides centralized logging configuration and utilities
for consistent logging across the entire application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from ..config import LOG_LEVEL, LOG_FORMAT


def setup_logger(
    name: str,
    level: Union[str, int] = LOG_LEVEL,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    file_output: bool = False
) -> logging.Logger:
    """Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if file_output is True)
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_file is None:
            # Create default log file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path("logs") / f"{name}_{timestamp}.log"
        
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up with defaults
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_execution_time(func):
    """Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_dataframe_info(df, name: str = "DataFrame", logger: Optional[logging.Logger] = None):
    """Log information about a DataFrame.
    
    Args:
        df: DataFrame to log info about
        name: Name to use in log messages
        logger: Logger instance (creates default if None)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {list(df.columns)}")
    logger.info(f"{name} memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if hasattr(df, 'index') and hasattr(df.index, 'min'):
        try:
            logger.info(f"{name} date range: {df.index.min()} to {df.index.max()}")
        except (AttributeError, TypeError):
            pass


def configure_third_party_loggers(level: Union[str, int] = "WARNING"):
    """Configure third-party library loggers to reduce noise.
    
    Args:
        level: Logging level for third-party loggers
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Common noisy loggers
    noisy_loggers = [
        "requests.packages.urllib3.connectionpool",
        "urllib3.connectionpool",
        "matplotlib.font_manager",
        "PIL.PngImagePlugin",
        "numba.core.ssa",
        "numba.core.interpreter",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(level)


# Set up default configuration when module is imported
configure_third_party_loggers()
