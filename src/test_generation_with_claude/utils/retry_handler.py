"""
Robust retry mechanism for MCP server connections and other operations.
"""
import asyncio
import time
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """Different retry strategies available."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, OSError)
    
class RetryHandler:
    """Advanced retry handler with multiple strategies."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self._fibonacci_cache = [1, 1]
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            while len(self._fibonacci_cache) < attempt:
                next_fib = self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
                self._fibonacci_cache.append(next_fib)
            delay = self.config.base_delay * self._fibonacci_cache[attempt - 1]
        
        # Apply jitter to prevent thundering herd
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return min(delay, self.config.max_delay)
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Success on attempt {attempt} for {func.__name__}")
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed for {func.__name__}: {str(e)}")
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed for {func.__name__}")
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception in {func.__name__}: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise last_exception
    
    async def async_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.info(f"Async attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Async success on attempt {attempt} for {func.__name__}")
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                logger.warning(f"Async attempt {attempt} failed for {func.__name__}: {str(e)}")
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Async retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} async attempts failed for {func.__name__}")
            
            except Exception as e:
                logger.error(f"Non-retryable async exception in {func.__name__}: {str(e)}")
                raise
        
        raise last_exception

def retry_on_failure(config: RetryConfig = None):
    """Decorator for automatic retry on function failure."""
    retry_config = config or RetryConfig()
    handler = RetryHandler(retry_config)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return handler.retry(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await handler.async_retry(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator

# MCP-specific retry configurations
MCP_CONNECTION_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError, RuntimeError)
)

MCP_OPERATION_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    strategy=RetryStrategy.LINEAR_BACKOFF,
    retryable_exceptions=(TimeoutError, ConnectionError)
)