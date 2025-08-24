"""
Comprehensive logging configuration for the CrewAI agent project.
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry)

class CrewAILogger:
    """Centralized logger configuration for CrewAI agent."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: str = "./logs",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 enable_json_logging: bool = False,
                 max_log_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.enable_json_logging = enable_json_logging
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        self._setup_root_logger()
        
        # Configure specific loggers
        self._setup_component_loggers()
    
    def _setup_root_logger(self):
        """Setup the root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            if self.enable_json_logging:
                console_handler.setFormatter(JsonFormatter())
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
            
            root_logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file_logging:
            self._setup_file_handlers(root_logger)
    
    def _setup_file_handlers(self, logger):
        """Setup rotating file handlers."""
        
        # General application log
        app_log_file = self.log_dir / "crewai_agent.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file, 
            maxBytes=self.max_log_size, 
            backupCount=self.backup_count
        )
        app_handler.setLevel(self.log_level)
        
        # Error log (errors and above only)
        error_log_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        
        # Performance log
        perf_log_file = self.log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        perf_handler.setLevel(logging.INFO)
        
        # MCP-specific log
        mcp_log_file = self.log_dir / "mcp_connections.log"
        mcp_handler = logging.handlers.RotatingFileHandler(
            mcp_log_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        mcp_handler.setLevel(logging.DEBUG)
        
        # Set formatters
        if self.enable_json_logging:
            json_formatter = JsonFormatter()
            app_handler.setFormatter(json_formatter)
            error_handler.setFormatter(json_formatter)
            perf_handler.setFormatter(json_formatter)
            mcp_handler.setFormatter(json_formatter)
        else:
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            app_handler.setFormatter(detailed_formatter)
            error_handler.setFormatter(detailed_formatter)
            perf_handler.setFormatter(detailed_formatter)
            mcp_handler.setFormatter(detailed_formatter)
        
        # Add handlers
        logger.addHandler(app_handler)
        logger.addHandler(error_handler)
        
        # Setup filtered loggers for specific components
        self._setup_filtered_logger("performance", perf_handler)
        self._setup_filtered_logger("mcp", mcp_handler)
    
    def _setup_filtered_logger(self, name: str, handler):
        """Setup a filtered logger for specific components."""
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.propagate = False
    
    def _setup_component_loggers(self):
        """Configure loggers for specific components."""
        
        # CrewAI components
        logging.getLogger("crewai").setLevel(self.log_level)
        logging.getLogger("crewai_tools").setLevel(self.log_level)
        
        # External libraries (reduce noise)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("boto3").setLevel(logging.WARNING)
        logging.getLogger("botocore").setLevel(logging.WARNING)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance for a specific component."""
        return logging.getLogger(name)
    
    @staticmethod
    def log_performance(operation: str, duration: float, success: bool, **kwargs):
        """Log performance metrics."""
        perf_logger = logging.getLogger("performance")
        perf_logger.info(
            f"Performance: {operation}",
            extra={
                'extra_fields': {
                    'operation': operation,
                    'duration_seconds': duration,
                    'success': success,
                    **kwargs
                }
            }
        )
    
    @staticmethod
    def log_mcp_event(event_type: str, server_name: str, status: str, **kwargs):
        """Log MCP-specific events."""
        mcp_logger = logging.getLogger("mcp")
        mcp_logger.info(
            f"MCP {event_type}: {server_name} - {status}",
            extra={
                'extra_fields': {
                    'event_type': event_type,
                    'server_name': server_name,
                    'status': status,
                    **kwargs
                }
            }
        )

# Context manager for performance logging
class PerformanceLogger:
    """Context manager for automatic performance logging."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, **kwargs):
        self.operation = operation
        self.logger = logger or logging.getLogger("performance")
        self.kwargs = kwargs
        self.start_time = None
        self.success = False
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        self.success = exc_type is None
        
        CrewAILogger.log_performance(
            self.operation, 
            duration, 
            self.success,
            **self.kwargs
        )
        
        if not self.success:
            self.logger.error(
                f"Operation failed: {self.operation}",
                exc_info=(exc_type, exc_val, exc_tb)
            )

# Initialize default logger configuration
def setup_logging(config: Dict[str, Any] = None) -> CrewAILogger:
    """Setup logging with optional configuration."""
    config = config or {}
    
    # Override with environment variables if available
    config.setdefault('log_level', os.getenv('LOG_LEVEL', 'INFO'))
    config.setdefault('log_dir', os.getenv('LOG_DIR', './logs'))
    config.setdefault('enable_json_logging', os.getenv('JSON_LOGGING', 'false').lower() == 'true')
    
    return CrewAILogger(**config)