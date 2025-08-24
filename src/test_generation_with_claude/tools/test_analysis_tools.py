"""
Enhanced test analysis and comparison tools for the API test generation workflow
"""

import os
import ast
import re
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from crewai.tools import BaseTool
from pydantic import Field


# Setup basic logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Custom exceptions
class ToolError(Exception):
    """Base exception for tool errors"""
    pass


class ValidationError(ToolError):
    """Exception for validation errors"""
    pass


class NetworkError(ToolError):
    """Exception for network-related errors"""
    pass


# Simple error handler class
class ErrorHandler:
    """Simple error handler for logging and tracking errors"""
    
    def __init__(self):
        self.errors = []
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None):
        """Handle and log an error with context"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "timestamp": time.time()
        }
        self.errors.append(error_info)
        logger.error(f"Error in {context.get('operation', 'unknown')}: {error}")


# Simple decorators
def handle_errors(error_class=Exception):
    """Decorator to handle errors and convert them to a specific exception type"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_class:
                raise
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise error_class(f"Error in {func.__name__}: {str(e)}") from e
        return wrapper
    return decorator


def retry_on_failure(max_attempts=3, delay=1):
    """Simple retry decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    logger.warning(f"Attempt {attempt + 1} of {max_attempts} failed: {e}")
            raise last_exception
        return wrapper
    return decorator


def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f} seconds: {e}")
            raise
    return wrapper


# Simple performance logger class
class PerformanceLogger:
    """Simple performance logger for tracking operation times"""
    
    def __init__(self):
        self.operations = []
    
    def log_operation(self, operation_name: str, duration: float):
        """Log an operation's performance"""
        self.operations.append({
            "operation": operation_name,
            "duration": duration,
            "timestamp": time.time()
        })
        logger.info(f"Operation '{operation_name}' took {duration:.2f} seconds")


# Simple retry handler class (placeholder for compatibility)
class RetryHandler:
    """Simple retry handler placeholder"""
    
    def __init__(self):
        self.retry_count = 0
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if operation should be retried"""
        return isinstance(error, (OSError, IOError))


@dataclass
class TestAnalysisMetrics:
    """Metrics for test analysis operations."""
    files_analyzed: int = 0
    tests_found: int = 0
    comparisons_made: int = 0
    coverage_calculations: int = 0
    errors_encountered: int = 0
    total_execution_time: float = 0.0


class TestComparisonTool(BaseTool):
    """Enhanced tool for comparing existing tests with requirements"""
    
    name: str = "TestComparisonTool"
    description: str = "Compare existing test files with extracted requirements to identify gaps with enhanced error handling and performance monitoring"
    
    # Declare these as optional fields to avoid Pydantic validation errors
    error_handler: Optional[ErrorHandler] = Field(default=None, exclude=True)
    retry_handler: Optional[RetryHandler] = Field(default=None, exclude=True)
    performance_logger: Optional[PerformanceLogger] = Field(default=None, exclude=True)
    metrics: Optional[TestAnalysisMetrics] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize these after parent __init__
        object.__setattr__(self, 'error_handler', ErrorHandler())
        object.__setattr__(self, 'retry_handler', RetryHandler())
        object.__setattr__(self, 'performance_logger', PerformanceLogger())
        object.__setattr__(self, 'metrics', TestAnalysisMetrics())
    
    @handle_errors(ToolError)
    @retry_on_failure(max_attempts=3)
    @log_performance
    def _run(self, test_files: List[str], requirements: Dict[str, Any], 
             test_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Compare existing tests with requirements with enhanced error handling
        
        Args:
            test_files: List of test file paths
            requirements: Dictionary of extracted requirements
            test_patterns: Optional patterns to look for in tests
        
        Returns:
            Dictionary with comparison results and identified gaps
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(test_files, requirements, test_patterns)
            
            if test_patterns is None:
                test_patterns = [
                    "test_",
                    "def test",
                    "@pytest.mark",
                    "assert",
                    "assertEqual",
                    "mock",
                    "fixture"
                ]
            
            comparison_result = {
                "success": True,
                "test_files_analyzed": [],
                "requirements_coverage": {},
                "test_gaps": [],
                "test_scenarios_found": [],
                "coverage_percentage": 0,
                "recommendations": [],
                "metrics": asdict(self.metrics),
                "timestamp": time.time()
            }
            
            # Analyze each test file with error handling
            for test_file in test_files:
                try:
                    file_analysis = self._analyze_test_file_safely(test_file)
                    comparison_result["test_files_analyzed"].append(file_analysis)
                    self.metrics.files_analyzed += 1
                    
                    # Count tests found
                    test_count = len(file_analysis.get("test_functions", []))
                    self.metrics.tests_found += test_count
                    
                except Exception as e:
                    self.metrics.errors_encountered += 1
                    self.error_handler.handle_error(e, {"file": test_file, "operation": "file_analysis"})
                    comparison_result["test_files_analyzed"].append({
                        "file": str(test_file),
                        "error": f"Failed to read file: {str(e)}",
                        "test_functions": [],
                        "test_scenarios": []
                    })
            
            # Compare with requirements
            requirements_coverage = self._compare_with_requirements_safely(
                comparison_result["test_files_analyzed"], 
                requirements
            )
            comparison_result.update(requirements_coverage)
            self.metrics.comparisons_made += 1
            
            # Log performance
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            self.performance_logger.log_operation("test_comparison", execution_time)
            
            return comparison_result
            
        except Exception as e:
            self.metrics.errors_encountered += 1
            self.error_handler.handle_error(e, {
                "operation": "test_comparison",
                "test_files": test_files,
                "requirements_keys": list(requirements.keys()) if isinstance(requirements, dict) else []
            })
            return {
                "success": False,
                "error": f"Comparison error: {str(e)}",
                "message": f"Failed to compare tests with requirements: {str(e)}",
                "metrics": asdict(self.metrics)
            }
    
    def _validate_inputs(self, test_files: List[str], requirements: Dict[str, Any], test_patterns: Optional[List[str]]):
        """Validate input parameters."""
        if not isinstance(test_files, list) or not test_files:
            raise ValidationError("test_files must be a non-empty list")
        
        if not isinstance(requirements, dict):
            raise ValidationError("requirements must be a dictionary")
        
        if test_patterns is not None and not isinstance(test_patterns, list):
            raise ValidationError("test_patterns must be a list or None")
    
    def _analyze_test_file_safely(self, test_file: str) -> Dict[str, Any]:
        """Safely analyze a test file with comprehensive error handling."""
        file_path = Path(test_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._analyze_test_file(content, str(file_path))
            
        except UnicodeDecodeError as e:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    return self._analyze_test_file(content, str(file_path))
                except UnicodeDecodeError:
                    continue
            
            raise ToolError(f"Could not decode file {test_file} with any supported encoding") from e
        
        except Exception as e:
            raise ToolError(f"Error reading test file {test_file}: {str(e)}") from e
    
    def _compare_with_requirements_safely(self, test_analysis: List[Dict], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Safely compare test analysis with requirements."""
        try:
            return self._compare_with_requirements(test_analysis, requirements)
        except Exception as e:
            self.error_handler.handle_error(e, {
                "operation": "requirements_comparison",
                "test_analysis_count": len(test_analysis),
                "requirements_keys": list(requirements.keys())
            })
            return {
                "requirements_coverage": [],
                "test_gaps": [{"error": f"Comparison failed: {str(e)}"}],
                "coverage_percentage": 0,
                "recommendations": ["Fix comparison errors before proceeding"],
                "total_requirements": 0,
                "covered_requirements": 0
            }
    
    def _analyze_test_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze a single test file to extract test information"""
        try:
            tree = ast.parse(content)
            
            test_functions = []
            test_scenarios = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        test_info = {
                            "name": node.name,
                            "line": node.lineno,
                            "docstring": ast.get_docstring(node),
                            "decorators": [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list],
                            "scenarios": self._extract_scenarios_from_function(node, content)
                        }
                        test_functions.append(test_info)
                        test_scenarios.extend(test_info["scenarios"])
                
                elif isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return {
                "file": file_path,
                "test_functions": test_functions,
                "test_scenarios": test_scenarios,
                "imports": imports,
                "function_count": len(test_functions),
                "has_fixtures": any("fixture" in imp for imp in imports),
                "has_mocking": any("mock" in imp.lower() for imp in imports),
                "has_parametrize": any("parametrize" in str(func.get("decorators", [])) for func in test_functions)
            }
            
        except SyntaxError:
            # Try to extract basic information without AST
            return self._extract_basic_test_info(content, file_path)
        except Exception as e:
            return {
                "file": file_path,
                "error": f"Analysis error: {str(e)}",
                "test_functions": [],
                "test_scenarios": []
            }
    
    def _extract_basic_test_info(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract basic test information when AST parsing fails"""
        test_functions = []
        
        # Use regex to find test functions
        test_pattern = re.compile(r'^def (test_\w+)', re.MULTILINE)
        matches = test_pattern.findall(content)
        
        for match in matches:
            test_functions.append({
                "name": match,
                "line": -1,
                "docstring": None,
                "decorators": [],
                "scenarios": []
            })
        
        return {
            "file": file_path,
            "test_functions": test_functions,
            "test_scenarios": [],
            "imports": [],
            "function_count": len(test_functions),
            "parsing_method": "regex_fallback"
        }
    
    def _extract_scenarios_from_function(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """Extract test scenarios from a function"""
        scenarios = []
        
        # Look for scenario descriptions in docstring
        docstring = ast.get_docstring(func_node)
        if docstring:
            # Look for scenario patterns
            scenario_patterns = [
                r"scenario[:\s]+(.+)",
                r"test[:\s]+(.+)",
                r"when[:\s]+(.+)",
                r"given[:\s]+(.+)",
                r"then[:\s]+(.+)"
            ]
            
            for pattern in scenario_patterns:
                matches = re.findall(pattern, docstring, re.IGNORECASE)
                scenarios.extend(matches)
        
        # Look for assert statements to understand what's being tested
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                # Try to extract what's being asserted
                if hasattr(node.test, 'left') and hasattr(node.test, 'comparators'):
                    scenarios.append(f"Assert condition in {func_node.name}")
        
        return scenarios
    
    def _compare_with_requirements(self, test_analysis: List[Dict], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Compare test analysis with requirements to find gaps"""
        test_scenarios = []
        test_functions = []
        
        # Collect all test scenarios and functions
        for file_analysis in test_analysis:
            test_scenarios.extend(file_analysis.get("test_scenarios", []))
            test_functions.extend([func["name"] for func in file_analysis.get("test_functions", [])])
        
        # Extract requirement keywords/topics
        requirement_topics = []
        if isinstance(requirements, dict):
            for key, value in requirements.items():
                if isinstance(value, str):
                    requirement_topics.extend(self._extract_keywords(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            requirement_topics.extend(self._extract_keywords(item))
        
        # Find coverage
        coverage_matches = []
        test_gaps = []
        
        for topic in requirement_topics:
            topic_covered = False
            for test_func in test_functions:
                if topic.lower() in test_func.lower():
                    coverage_matches.append({
                        "requirement_topic": topic,
                        "covered_by": test_func
                    })
                    topic_covered = True
                    break
            
            if not topic_covered:
                test_gaps.append({
                    "missing_topic": topic,
                    "suggestion": f"Add test for {topic} functionality"
                })
        
        coverage_percentage = (len(coverage_matches) / max(len(requirement_topics), 1)) * 100
        
        recommendations = []
        if coverage_percentage < 70:
            recommendations.append("Consider adding more comprehensive test coverage")
        if not any("integration" in func.lower() for func in test_functions):
            recommendations.append("Consider adding integration tests")
        if not any("error" in func.lower() or "exception" in func.lower() for func in test_functions):
            recommendations.append("Consider adding error handling tests")
        
        return {
            "requirements_coverage": coverage_matches,
            "test_gaps": test_gaps,
            "coverage_percentage": round(coverage_percentage, 2),
            "recommendations": recommendations,
            "total_requirements": len(requirement_topics),
            "covered_requirements": len(coverage_matches)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from requirement text"""
        # Simple keyword extraction
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))  # Remove duplicates


class TestCoverageAnalyzerTool(BaseTool):
    """Enhanced tool for analyzing test coverage and identifying gaps"""
    
    name: str = "TestCoverageAnalyzerTool"
    description: str = "Analyze test coverage for API endpoints and identify missing test scenarios with robust error handling"
    
    # Declare these as optional fields to avoid Pydantic validation errors
    error_handler: Optional[ErrorHandler] = Field(default=None, exclude=True)
    retry_handler: Optional[RetryHandler] = Field(default=None, exclude=True)
    performance_logger: Optional[PerformanceLogger] = Field(default=None, exclude=True)
    metrics: Optional[TestAnalysisMetrics] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize these after parent __init__
        object.__setattr__(self, 'error_handler', ErrorHandler())
        object.__setattr__(self, 'retry_handler', RetryHandler())
        object.__setattr__(self, 'performance_logger', PerformanceLogger())
        object.__setattr__(self, 'metrics', TestAnalysisMetrics())
    
    @handle_errors(ToolError)
    @retry_on_failure(max_attempts=3)
    @log_performance
    def _run(self, backend_files: List[str], test_files: List[str], 
             framework: str = "fastapi") -> Dict[str, Any]:
        """
        Analyze test coverage for backend API endpoints with enhanced error handling
        
        Args:
            backend_files: List of backend source files
            test_files: List of test files
            framework: Backend framework (fastapi/django)
        
        Returns:
            Dictionary with coverage analysis results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_coverage_inputs(backend_files, test_files, framework)
            
            coverage_result = {
                "success": True,
                "api_endpoints": [],
                "tested_endpoints": [],
                "untested_endpoints": [],
                "coverage_percentage": 0,
                "missing_test_types": [],
                "recommendations": [],
                "metrics": asdict(self.metrics),
                "timestamp": time.time()
            }
            
            # Extract API endpoints from backend files with error handling
            api_endpoints = self._extract_api_endpoints_safely(backend_files, framework)
            coverage_result["api_endpoints"] = api_endpoints
            
            # Extract tested endpoints from test files with error handling
            tested_endpoints = self._extract_tested_endpoints_safely(test_files)
            coverage_result["tested_endpoints"] = tested_endpoints
            
            # Find untested endpoints
            untested = []
            for endpoint in api_endpoints:
                if not any(self._endpoint_matches_test(endpoint, test) for test in tested_endpoints):
                    untested.append(endpoint)
            
            coverage_result["untested_endpoints"] = untested
            
            # Calculate coverage percentage
            total_endpoints = len(api_endpoints)
            tested_count = total_endpoints - len(untested)
            coverage_percentage = (tested_count / max(total_endpoints, 1)) * 100
            coverage_result["coverage_percentage"] = round(coverage_percentage, 2)
            self.metrics.coverage_calculations += 1
            
            # Identify missing test types
            missing_types = self._identify_missing_test_types(api_endpoints, tested_endpoints)
            coverage_result["missing_test_types"] = missing_types
            
            # Generate recommendations
            recommendations = self._generate_coverage_recommendations(coverage_result)
            coverage_result["recommendations"] = recommendations
            
            # Log performance
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            self.performance_logger.log_operation("coverage_analysis", execution_time)
            
            return coverage_result
            
        except Exception as e:
            self.metrics.errors_encountered += 1
            self.error_handler.handle_error(e, {
                "operation": "coverage_analysis",
                "backend_files_count": len(backend_files),
                "test_files_count": len(test_files),
                "framework": framework
            })
            return {
                "success": False,
                "error": f"Coverage analysis error: {str(e)}",
                "message": f"Failed to analyze test coverage: {str(e)}",
                "metrics": asdict(self.metrics)
            }
    
    def _validate_coverage_inputs(self, backend_files: List[str], test_files: List[str], framework: str):
        """Validate coverage analysis inputs."""
        if not isinstance(backend_files, list) or not backend_files:
            raise ValidationError("backend_files must be a non-empty list")
        
        if not isinstance(test_files, list) or not test_files:
            raise ValidationError("test_files must be a non-empty list")
        
        if framework.lower() not in ['fastapi', 'django', 'flask']:
            raise ValidationError(f"Unsupported framework: {framework}")
    
    def _extract_api_endpoints_safely(self, backend_files: List[str], framework: str) -> List[Dict[str, Any]]:
        """Safely extract API endpoints from backend files."""
        endpoints = []
        
        for file_path in backend_files:
            try:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    self.error_handler.handle_error(
                        FileNotFoundError(f"Backend file not found: {file_path}"),
                        {"operation": "endpoint_extraction", "file": file_path}
                    )
                    continue
                
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if framework.lower() == "fastapi":
                    file_endpoints = self._extract_fastapi_endpoints(content, file_path)
                elif framework.lower() == "django":
                    file_endpoints = self._extract_django_endpoints(content, file_path)
                elif framework.lower() == "flask":
                    file_endpoints = self._extract_flask_endpoints(content, file_path)
                else:
                    file_endpoints = []
                
                endpoints.extend(file_endpoints)
                self.metrics.files_analyzed += 1
                    
            except Exception as e:
                self.metrics.errors_encountered += 1
                self.error_handler.handle_error(e, {
                    "operation": "endpoint_extraction",
                    "file": file_path,
                    "framework": framework
                })
                continue
        
        return endpoints
    
    def _extract_tested_endpoints_safely(self, test_files: List[str]) -> List[Dict[str, Any]]:
        """Safely extract tested endpoints from test files."""
        tested_endpoints = []
        
        for file_path in test_files:
            try:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    continue
                
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_tested = self._extract_tested_endpoints_from_content(content, file_path)
                tested_endpoints.extend(file_tested)
                self.metrics.files_analyzed += 1
                        
            except Exception as e:
                self.metrics.errors_encountered += 1
                self.error_handler.handle_error(e, {
                    "operation": "tested_endpoint_extraction",
                    "file": file_path
                })
                continue
        
        return tested_endpoints
    
    def _extract_flask_endpoints(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract Flask endpoints from source code."""
        endpoints = []
        
        # Pattern for Flask route decorators
        patterns = [
            r'@app\.route\(["\']([^"\']+)["\'].*methods=\[([^\]]+)\]',
            r'@app\.route\(["\']([^"\']+)["\']',
            r'@\w+\.route\(["\']([^"\']+)["\'].*methods=\[([^\]]+)\]',
            r'@\w+\.route\(["\']([^"\']+)["\']'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2 and match[1]:  # Has methods
                    path, methods_str = match
                    methods = re.findall(r'["\'](\w+)["\']', methods_str)
                    for method in methods:
                        endpoints.append({
                            "method": method.upper(),
                            "path": path,
                            "file": file_path,
                            "framework": "flask"
                        })
                else:  # No explicit methods, assume GET
                    path = match[0] if isinstance(match, tuple) else match
                    endpoints.append({
                        "method": "GET",
                        "path": path,
                        "file": file_path,
                        "framework": "flask"
                    })
        
        return endpoints
    
    def _extract_tested_endpoints_from_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract tested endpoints from test file content."""
        tested_endpoints = []
        
        # Look for HTTP method calls in tests
        patterns = [
            r'client\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            r'requests\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            r'self\.client\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            r'test_client\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for method, path in matches:
                tested_endpoints.append({
                    "method": method.upper(),
                    "path": path,
                    "test_file": file_path
                })        
        return tested_endpoints
    
    # Legacy method redirects for backward compatibility
    def _extract_api_endpoints(self, backend_files: List[str], framework: str) -> List[Dict[str, Any]]:
        """Legacy method - redirects to enhanced version."""
        return self._extract_api_endpoints_safely(backend_files, framework)
    
    def _extract_tested_endpoints(self, test_files: List[str]) -> List[Dict[str, Any]]:
        """Legacy method - redirects to enhanced version."""
        return self._extract_tested_endpoints_safely(test_files)
    
    def _extract_fastapi_endpoints(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract FastAPI endpoints from source code"""
        endpoints = []
        
        # Pattern for FastAPI route decorators
        patterns = [
            r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for method, path in matches:
                endpoints.append({
                    "method": method.upper(),
                    "path": path,
                    "file": file_path,
                    "framework": "fastapi"
                })
        
        return endpoints
    
    def _extract_django_endpoints(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract Django endpoints from source code"""
        endpoints = []
        
        # Look for Django URL patterns
        url_patterns = re.findall(r'path\(["\']([^"\']+)["\']', content)
        
        for path in url_patterns:
            # Django paths might not have explicit methods, assume common ones
            for method in ["GET", "POST", "PUT", "DELETE"]:
                endpoints.append({
                    "method": method,
                    "path": path,
                    "file": file_path,
                    "framework": "django"
                })
        return endpoints
    
    def _endpoint_matches_test(self, endpoint: Dict[str, Any], test: Dict[str, Any]) -> bool:
        """Check if an endpoint is covered by a test"""
        return (endpoint["method"] == test["method"] and 
                self._paths_match(endpoint["path"], test["path"]))
    
    def _paths_match(self, endpoint_path: str, test_path: str) -> bool:
        """Check if endpoint path matches test path (considering path parameters)"""
        # Simple matching - could be enhanced for path parameters
        return endpoint_path == test_path or endpoint_path in test_path or test_path in endpoint_path
    
    def _identify_missing_test_types(self, endpoints: List[Dict], tested: List[Dict]) -> List[str]:
        """Identify missing test types"""
        missing_types = []
        
        # Check for different HTTP methods
        endpoint_methods = set(ep["method"] for ep in endpoints)
        tested_methods = set(test["method"] for test in tested)
        
        for method in endpoint_methods:
            if method not in tested_methods:
                missing_types.append(f"Missing {method} method tests")
        
        # Check for error handling tests
        error_patterns = ["400", "401", "403", "404", "500", "error", "exception"]
        has_error_tests = any(any(pattern in str(test).lower() for pattern in error_patterns) for test in tested)
        
        if not has_error_tests:
            missing_types.append("Missing error handling tests")
        
        return missing_types
    
    def _generate_coverage_recommendations(self, coverage_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on coverage analysis"""
        recommendations = []
        
        coverage_pct = coverage_result["coverage_percentage"]
        
        if coverage_pct < 50:
            recommendations.append("Critical: Test coverage is very low. Consider adding comprehensive test suite.")
        elif coverage_pct < 80:
            recommendations.append("Test coverage needs improvement. Focus on untested endpoints.")
        
        untested_count = len(coverage_result["untested_endpoints"])
        if untested_count > 0:
            recommendations.append(f"Add tests for {untested_count} untested endpoints.")
        
        if coverage_result["missing_test_types"]:
            recommendations.append("Add missing test types: " + ", ".join(coverage_result["missing_test_types"]))
        
        return recommendations


class GapAnalyzerTool(BaseTool):
    """Enhanced tool for identifying gaps between requirements and existing tests"""
    
    name: str = "GapAnalyzerTool"
    description: str = "Analyze gaps between Jira requirements and existing test coverage with enhanced error handling"
    
    # Declare these as optional fields to avoid Pydantic validation errors
    error_handler: Optional[ErrorHandler] = Field(default=None, exclude=True)
    retry_handler: Optional[RetryHandler] = Field(default=None, exclude=True)
    performance_logger: Optional[PerformanceLogger] = Field(default=None, exclude=True)
    metrics: Optional[TestAnalysisMetrics] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize these after parent __init__
        object.__setattr__(self, 'error_handler', ErrorHandler())
        object.__setattr__(self, 'retry_handler', RetryHandler())
        object.__setattr__(self, 'performance_logger', PerformanceLogger())
        object.__setattr__(self, 'metrics', TestAnalysisMetrics())
    
    @handle_errors(ToolError)
    @retry_on_failure(max_attempts=2)
    @log_performance
    def _run(self, requirements: Dict[str, Any], test_analysis: Dict[str, Any], 
             coverage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze gaps between requirements and current test state with enhanced error handling
        
        Args:
            requirements: Extracted requirements from Jira
            test_analysis: Results from test comparison analysis
            coverage_analysis: Results from coverage analysis
        
        Returns:
            Dictionary with comprehensive gap analysis
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_gap_inputs(requirements, test_analysis, coverage_analysis)
            
            gap_result = {
                "success": True,
                "functional_gaps": [],
                "coverage_gaps": [],
                "quality_gaps": [],
                "priority_gaps": [],
                "recommendations": [],
                "gap_score": 0,
                "metrics": asdict(self.metrics),
                "timestamp": time.time()
            }
            
            # Analyze functional gaps with error handling
            functional_gaps = self._analyze_functional_gaps_safely(requirements, test_analysis)
            gap_result["functional_gaps"] = functional_gaps
            
            # Analyze coverage gaps with error handling
            coverage_gaps = self._analyze_coverage_gaps_safely(coverage_analysis)
            gap_result["coverage_gaps"] = coverage_gaps
            
            # Analyze quality gaps with error handling
            quality_gaps = self._analyze_quality_gaps_safely(test_analysis)
            gap_result["quality_gaps"] = quality_gaps
              # Prioritize gaps
            priority_gaps = self._prioritize_gaps(functional_gaps, coverage_gaps, quality_gaps)
            gap_result["priority_gaps"] = priority_gaps
            
            # Calculate overall gap score
            gap_score = self._calculate_gap_score(gap_result)
            gap_result["gap_score"] = gap_score
            
            # Generate recommendations
            recommendations = self._generate_gap_recommendations(gap_result)
            gap_result["recommendations"] = recommendations
            
            # Log performance
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            self.performance_logger.log_operation("gap_analysis", execution_time)
            
            return gap_result
            
        except Exception as e:
            self.metrics.errors_encountered += 1
            self.error_handler.handle_error(e, {
                "operation": "gap_analysis",
                "requirements_keys": list(requirements.keys()) if isinstance(requirements, dict) else [],
                "test_analysis_success": test_analysis.get("success", False),
                "coverage_analysis_success": coverage_analysis.get("success", False)
            })
            return {
                "success": False,
                "error": f"Gap analysis error: {str(e)}",
                "message": f"Failed to analyze gaps: {str(e)}",
                "metrics": asdict(self.metrics)
            }
    
    def _validate_gap_inputs(self, requirements: Dict[str, Any], test_analysis: Dict[str, Any], coverage_analysis: Dict[str, Any]):
        """Validate gap analysis inputs."""
        if not isinstance(requirements, dict):
            raise ValidationError("requirements must be a dictionary")
        
        if not isinstance(test_analysis, dict):
            raise ValidationError("test_analysis must be a dictionary")
        
        if not isinstance(coverage_analysis, dict):
            raise ValidationError("coverage_analysis must be a dictionary")
    
    def _analyze_functional_gaps_safely(self, requirements: Dict[str, Any], test_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Safely analyze functional requirement gaps."""
        try:
            return self._analyze_functional_gaps(requirements, test_analysis)
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "functional_gaps_analysis"})
            return [{"error": f"Functional gap analysis failed: {str(e)}"}]
    
    def _analyze_coverage_gaps_safely(self, coverage_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Safely analyze endpoint coverage gaps."""
        try:
            return self._analyze_coverage_gaps(coverage_analysis)
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "coverage_gaps_analysis"})
            return [{"error": f"Coverage gap analysis failed: {str(e)}"}]
    
    def _analyze_quality_gaps_safely(self, test_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Safely analyze test quality gaps."""
        try:
            return self._analyze_quality_gaps(test_analysis)
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "quality_gaps_analysis"})
            return [{"error": f"Quality gap analysis failed: {str(e)}"}]
    
    def _analyze_functional_gaps(self, requirements: Dict[str, Any], test_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze functional requirement gaps"""
        gaps = []
        
        req_topics = []
        if isinstance(requirements, dict):
            for key, value in requirements.items():
                if isinstance(value, str):
                    req_topics.extend(self._extract_functional_requirements(value))
        
        test_coverage = test_analysis.get("requirements_coverage", [])
        covered_topics = [item.get("requirement_topic", "") for item in test_coverage]
        
        for topic in req_topics:
            if topic not in covered_topics:
                gaps.append({
                    "type": "functional",
                    "requirement": topic,
                    "severity": "high" if self._is_critical_requirement(topic) else "medium",
                    "description": f"Missing test coverage for {topic}"
                })
        
        return gaps
    
    def _analyze_coverage_gaps(self, coverage_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze endpoint coverage gaps"""
        gaps = []
        
        untested_endpoints = coverage_analysis.get("untested_endpoints", [])
        
        for endpoint in untested_endpoints:
            gaps.append({
                "type": "coverage",
                "endpoint": f"{endpoint['method']} {endpoint['path']}",
                "severity": "high" if endpoint["method"] in ["POST", "DELETE"] else "medium",
                "description": f"No tests found for {endpoint['method']} {endpoint['path']}"
            })
        
        return gaps
    
    def _analyze_quality_gaps(self, test_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze test quality gaps"""
        gaps = []
        
        # Check for missing test patterns
        test_files = test_analysis.get("test_files_analyzed", [])
        
        has_integration_tests = any("integration" in str(file_info).lower() for file_info in test_files)
        has_error_tests = any("error" in str(file_info).lower() or "exception" in str(file_info).lower() for file_info in test_files)
        has_mocking = any(file_info.get("has_mocking", False) for file_info in test_files)
        
        if not has_integration_tests:
            gaps.append({
                "type": "quality",
                "category": "integration_tests",
                "severity": "medium",
                "description": "Missing integration tests"
            })
        
        if not has_error_tests:
            gaps.append({
                "type": "quality", 
                "category": "error_handling",
                "severity": "high",
                "description": "Missing error handling tests"
            })
        
        if not has_mocking:
            gaps.append({
                "type": "quality",
                "category": "mocking",
                "severity": "low",
                "description": "Consider adding mocking for external dependencies"
            })
        
        return gaps
    
    def _prioritize_gaps(self, functional_gaps: List, coverage_gaps: List, quality_gaps: List) -> List[Dict[str, Any]]:
        """Prioritize gaps by severity and impact"""
        all_gaps = functional_gaps + coverage_gaps + quality_gaps
        
        priority_order = {"high": 3, "medium": 2, "low": 1}
        
        sorted_gaps = sorted(all_gaps, key=lambda x: priority_order.get(x.get("severity", "low"), 1), reverse=True)
        
        return sorted_gaps[:10]  # Top 10 priority gaps
    
    def _calculate_gap_score(self, gap_result: Dict[str, Any]) -> float:
        """Calculate an overall gap score (0-100, lower is better)"""
        total_gaps = (len(gap_result["functional_gaps"]) + 
                     len(gap_result["coverage_gaps"]) + 
                     len(gap_result["quality_gaps"]))
        
        high_severity_gaps = sum(1 for gap in gap_result["priority_gaps"] if gap.get("severity") == "high")
        
        # Simple scoring: base score + severity weighting
        base_score = min(total_gaps * 5, 70)  # Max 70 from count
        severity_penalty = high_severity_gaps * 10  # 10 points per high severity
        
        gap_score = min(base_score + severity_penalty, 100)
        return round(gap_score, 2)
    
    def _generate_gap_recommendations(self, gap_result: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        gap_score = gap_result["gap_score"]
        
        if gap_score > 80:
            recommendations.append("Critical: Major test gaps identified. Comprehensive test suite needed.")
        elif gap_score > 50:
            recommendations.append("Significant gaps found. Focus on high-priority areas first.")
        elif gap_score > 20:
            recommendations.append("Minor gaps identified. Consider addressing for completeness.")
        else:
            recommendations.append("Good test coverage. Minor improvements possible.")
        
        # Add specific recommendations based on gap types
        priority_gaps = gap_result["priority_gaps"]
        
        functional_count = sum(1 for gap in priority_gaps if gap["type"] == "functional")
        coverage_count = sum(1 for gap in priority_gaps if gap["type"] == "coverage")
        quality_count = sum(1 for gap in priority_gaps if gap["type"] == "quality")
        
        if functional_count > 0:
            recommendations.append(f"Address {functional_count} functional requirement gaps")
        if coverage_count > 0:
            recommendations.append(f"Add tests for {coverage_count} untested endpoints")
        if quality_count > 0:
            recommendations.append(f"Improve test quality in {quality_count} areas")
        
        return recommendations
    
    def _extract_functional_requirements(self, text: str) -> List[str]:
        """Extract functional requirements from text"""
        # Simple extraction - could be enhanced with NLP
        functional_keywords = ["api", "endpoint", "authentication", "authorization", "validation", "crud", "database"]
        requirements = []        
        for keyword in functional_keywords:
            if keyword.lower() in text.lower():
                requirements.append(keyword)
        
        return requirements
    
    def _is_critical_requirement(self, requirement: str) -> bool:
        """Determine if a requirement is critical"""
        critical_keywords = ["authentication", "authorization", "security", "validation", "crud"]
        return any(keyword in requirement.lower() for keyword in critical_keywords)


# Backward compatibility aliases
TestComparison = TestComparisonTool
TestCoverageAnalyzer = TestCoverageAnalyzerTool
GapAnalyzer = GapAnalyzerTool