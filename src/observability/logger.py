"""Structured logging for the ML pipeline."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


class StructuredLogger:
    """Structured logger with JSON output."""
    
    def __init__(
        self,
        name: str = "ml_pipeline",
        log_level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_file: Optional log file path
        """
        self.name = name
        self.log_level = log_level
        self.log_file = log_file
        
        # Setup structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Get logger
        self.logger = structlog.get_logger(name)
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        # Get stdlib logger
        stdlib_logger = logging.getLogger(self.name)
        stdlib_logger.setLevel(getattr(logging, self.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))
        stdlib_logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(getattr(logging, self.log_level))
            stdlib_logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)


# Global logger instance
_default_logger = None


def get_logger(
    name: str = "ml_pipeline",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> StructuredLogger:
    """
    Get or create logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        StructuredLogger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = StructuredLogger(name, log_level, log_file)
    return _default_logger
