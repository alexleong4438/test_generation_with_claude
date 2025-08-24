"""
Comprehensive error classification and recovery mechanisms.
"""
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Callable, Type, List
from dataclasses import dataclass
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    DATA_VALIDATION = "data_validation"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"
    USER_INPUT = "user_input"

@dataclass
class ErrorInfo:
    """Detailed error information."""
    exception: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: float
    recoverable: bool = True
    suggested_action: Optional[str] = None

class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies."""
    
    @abstractmethod
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Check if this strategy can handle the error."""
        pass
    
    @abstractmethod
    def recover(self, error_info: ErrorInfo, **kwargs) -> Any:
        """Attempt to recover from the error."""
        pass

class RetryRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy using retry mechanism."""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return error_info.category in [
            ErrorCategory.CONNECTION,
            ErrorCategory.TIMEOUT,
            ErrorCategory.EXTERNAL_SERVICE
        ] and error_info.severity != ErrorSeverity.CRITICAL
    
    def recover(self, error_info: ErrorInfo, func: Callable, *args, **kwargs) -> Any:
        """Retry the operation."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Recovery attempt {attempt + 1}/{self.max_retries}")
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Recovery attempt {attempt + 1} failed: {e}")
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
        return None

class FallbackRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy using fallback mechanisms."""
    
    def __init__(self, fallback_func: Optional[Callable] = None):
        self.fallback_func = fallback_func
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return (error_info.category == ErrorCategory.EXTERNAL_SERVICE and 
                self.fallback_func is not None)
    
    def recover(self, error_info: ErrorInfo, **kwargs) -> Any:
        """Use fallback mechanism."""
        logger.info("Using fallback recovery strategy")
        return self.fallback_func(**kwargs)

class ResetRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy that resets connections or resources."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return error_info.category in [
            ErrorCategory.CONNECTION,
            ErrorCategory.RESOURCE_UNAVAILABLE
        ]
    
    def recover(self, error_info: ErrorInfo, reset_func: Callable, **kwargs) -> Any:
        """Reset the resource and retry."""
        logger.info("Attempting resource reset recovery")
        reset_func()
        return True

class ErrorClassifier:
    """Classifies errors into categories and severity levels."""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
    
    def _build_classification_rules(self) -> Dict[Type[Exception], tuple]:
        """Build rules for error classification."""
        return {
            ConnectionError: (ErrorCategory.CONNECTION, ErrorSeverity.HIGH),
            TimeoutError: (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
            PermissionError: (ErrorCategory.PERMISSION, ErrorSeverity.HIGH),
            FileNotFoundError: (ErrorCategory.RESOURCE_UNAVAILABLE, ErrorSeverity.MEDIUM),
            ValueError: (ErrorCategory.DATA_VALIDATION, ErrorSeverity.LOW),
            KeyError: (ErrorCategory.DATA_VALIDATION, ErrorSeverity.LOW),
            TypeError: (ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM),
            ImportError: (ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH),
            OSError: (ErrorCategory.RESOURCE_UNAVAILABLE, ErrorSeverity.MEDIUM),
            RuntimeError: (ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM),
        }
    
    def classify(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Classify an exception into an ErrorInfo object."""
        context = context or {}
        
        # Get base classification
        exc_type = type(exception)
        category, severity = self.classification_rules.get(
            exc_type, 
            (ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM)
        )
        
        # Refine classification based on exception message and context
        category, severity = self._refine_classification(
            exception, category, severity, context
        )
        
        # Determine if error is recoverable
        recoverable = self._is_recoverable(category, severity, exception)
        
        # Generate suggested action
        suggested_action = self._get_suggested_action(category, exception)
        
        return ErrorInfo(
            exception=exception,
            category=category,
            severity=severity,
            message=str(exception),
            context=context,
            timestamp=time.time(),
            recoverable=recoverable,
            suggested_action=suggested_action
        )
    
    def _refine_classification(self, exception: Exception, 
                             category: ErrorCategory, 
                             severity: ErrorSeverity,
                             context: Dict[str, Any]) -> tuple:
        """Refine classification based on additional context."""
        message = str(exception).lower()
        
        # Connection-related refinements
        if any(keyword in message for keyword in ['connection', 'network', 'socket']):
            category = ErrorCategory.CONNECTION
            severity = ErrorSeverity.HIGH
        
        # Authentication refinements
        elif any(keyword in message for keyword in ['auth', 'credential', 'token', 'unauthorized']):
            category = ErrorCategory.AUTHENTICATION
            severity = ErrorSeverity.HIGH
        
        # Timeout refinements
        elif any(keyword in message for keyword in ['timeout', 'deadline']):
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.MEDIUM
        
        # MCP-specific refinements
        if context.get('component') == 'mcp':
            if category == ErrorCategory.CONNECTION:
                severity = ErrorSeverity.CRITICAL
        
        return category, severity
    
    def _is_recoverable(self, category: ErrorCategory, 
                       severity: ErrorSeverity, 
                       exception: Exception) -> bool:
        """Determine if an error is recoverable."""
        # Critical errors are generally not recoverable
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        # Some categories are generally recoverable
        recoverable_categories = [
            ErrorCategory.CONNECTION,
            ErrorCategory.TIMEOUT,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.RESOURCE_UNAVAILABLE
        ]
        
        return category in recoverable_categories
    
    def _get_suggested_action(self, category: ErrorCategory, 
                            exception: Exception) -> str:
        """Generate suggested action for error recovery."""
        suggestions = {
            ErrorCategory.CONNECTION: "Check network connectivity and retry",
            ErrorCategory.AUTHENTICATION: "Verify credentials and refresh tokens",
            ErrorCategory.CONFIGURATION: "Review configuration settings",
            ErrorCategory.TIMEOUT: "Increase timeout values or check service availability",
            ErrorCategory.RESOURCE_UNAVAILABLE: "Check resource availability and permissions",
            ErrorCategory.DATA_VALIDATION: "Validate input data format and values",
            ErrorCategory.PERMISSION: "Check file/resource permissions",
            ErrorCategory.EXTERNAL_SERVICE: "Check external service status and retry",
        }
        
        return suggestions.get(category, "Review logs and contact support")

class ErrorRecoveryManager:
    """Manages error recovery strategies and execution."""
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.error_history: List[ErrorInfo] = []
        self.max_history = 100
    
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy."""
        self.recovery_strategies.append(strategy)
    
    def handle_error(self, exception: Exception, 
                    context: Dict[str, Any] = None,
                    recovery_func: Optional[Callable] = None,
                    **recovery_kwargs) -> Optional[Any]:
        """Handle an error with classification and recovery."""
        # Classify the error
        error_info = self.classifier.classify(exception, context)
        
        # Log the error
        self._log_error(error_info)
        
        # Store in history
        self._store_error(error_info)
        
        # Attempt recovery if error is recoverable
        if error_info.recoverable and recovery_func:
            return self._attempt_recovery(error_info, recovery_func, **recovery_kwargs)
        
        # Re-raise if not recoverable
        logger.error(f"Error not recoverable: {error_info.message}")
        raise exception
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error information."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error_info.severity]
        
        logger.log(
            log_level,
            f"Error classified - Category: {error_info.category.value}, "
            f"Severity: {error_info.severity.value}, "
            f"Message: {error_info.message}",
            extra={
                'extra_fields': {
                    'error_category': error_info.category.value,
                    'error_severity': error_info.severity.value,
                    'error_recoverable': error_info.recoverable,
                    'suggested_action': error_info.suggested_action,
                    'context': error_info.context
                }
            },
            exc_info=error_info.exception
        )
    
    def _store_error(self, error_info: ErrorInfo):
        """Store error in history."""
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
    
    def _attempt_recovery(self, error_info: ErrorInfo, 
                         recovery_func: Callable, 
                         **recovery_kwargs) -> Optional[Any]:
        """Attempt recovery using available strategies."""
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_info):
                try:
                    logger.info(f"Attempting recovery with {strategy.__class__.__name__}")
                    return strategy.recover(
                        error_info, 
                        func=recovery_func, 
                        **recovery_kwargs
                    )
                except Exception as recovery_exception:
                    logger.warning(
                        f"Recovery strategy {strategy.__class__.__name__} failed: "
                        f"{recovery_exception}"
                    )
        
        logger.error("All recovery strategies failed")
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics from history."""
        if not self.error_history:
            return {}
        
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'category_distribution': category_counts,
            'severity_distribution': severity_counts,
            'recoverable_percentage': sum(1 for e in self.error_history if e.recoverable) / len(self.error_history) * 100
        }

@contextmanager
def error_handling_context(context: Dict[str, Any] = None, 
                          recovery_manager: ErrorRecoveryManager = None):
    """Context manager for comprehensive error handling."""
    recovery_manager = recovery_manager or ErrorRecoveryManager()
    context = context or {}
    
    try:
        yield recovery_manager
    except Exception as e:
        recovery_manager.handle_error(e, context)
        raise

# Global error recovery manager instance
global_recovery_manager = ErrorRecoveryManager()

# Add default recovery strategies
global_recovery_manager.add_recovery_strategy(RetryRecoveryStrategy())
global_recovery_manager.add_recovery_strategy(ResetRecoveryStrategy())