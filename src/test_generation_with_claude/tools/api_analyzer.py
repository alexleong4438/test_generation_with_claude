"""
Enhanced API Analyzer Tools - OpenAPI Parser and FastAPI Analyzer
"""

import ast
import os
import json
import time
import requests
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from urllib.parse import urlparse
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict


# Simple error handling utilities
class ToolError(Exception):
    """Custom exception for tool errors"""
    pass


class ValidationError(ValueError):
    """Custom validation error"""
    pass


class NetworkError(ConnectionError):
    """Custom network error"""
    pass


class OpenAPISpecInput(BaseModel):
    """Input model for OpenAPI specification reading"""
    spec_path: str = Field(..., description="Path to local file or URL to OpenAPI spec (JSON/YAML)")

# Simple decorators for error handling
def handle_errors(error_class):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {str(e)}")
                raise error_class(str(e)) from e
        return wrapper
    return decorator


def retry_on_failure(max_attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            return None
        return wrapper
    return decorator


def log_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"{func.__name__} executed in {execution_time:.3f} seconds")
        return result
    return wrapper


@dataclass
class APIAnalysisMetrics:
    """Metrics for API analysis operations."""
    files_analyzed: int = 0
    endpoints_found: int = 0
    schemas_extracted: int = 0
    errors_encountered: int = 0
    total_execution_time: float = 0.0


class APIEndpoint(BaseModel):
    """Comprehensive model for API endpoint information"""
    # Basic endpoint info
    path: str
    method: str
    handler_function: str
    description: Optional[str] = None
    summary: Optional[str] = None
    
    # Request information
    request_schema: Optional[Dict[str, Any]] = None
    parameters: Optional[List[Dict[str, Any]]] = None  # Path, query, header params
    request_body: Optional[Dict[str, Any]] = None
    content_types: Optional[List[str]] = None  # Accepted content types
    
    # Response information
    response_schemas: Optional[Dict[str, Dict[str, Any]]] = None  # Multiple status codes
    response_content_types: Optional[List[str]] = None
    
    # Security and authentication
    auth_required: bool = False
    security_schemes: Optional[List[Dict[str, Any]]] = None
    
    # Validation and constraints
    path_parameters: Optional[List[Dict[str, Any]]] = None
    query_parameters: Optional[List[Dict[str, Any]]] = None
    header_parameters: Optional[List[Dict[str, Any]]] = None
    
    # Test generation metadata
    tags: Optional[List[str]] = None  # For test organization
    deprecated: bool = False
    external_docs: Optional[Dict[str, Any]] = None
    examples: Optional[Dict[str, Any]] = None  # Request/response examples
    
    # Server and environment info
    servers: Optional[List[Dict[str, Any]]] = None


class OpenAPISpecReaderTool(BaseTool):
    """
        Tool to read OpenAPI/Swagger specifications from local files or URLs
        Returns:
            OpenAPI specification content in JSON format
    """
    
    name: str = "OpenAPI Spec Reader"
    description: str = """
        Reads OpenAPI/Swagger specifications (JSON/YAML) from local files or URLs and returns the content in JSON format
    """
    args_schema: Type[BaseModel] = OpenAPISpecInput

    def __init__(self):
        super().__init__()
        self._metrics = APIAnalysisMetrics()

    @handle_errors(ToolError)
    @retry_on_failure(max_attempts=3)
    @log_performance
    def _run(self, spec_path: str) -> str:
        """
        Read OpenAPI specification from local file or URL and return as JSON
        
        Args:
            spec_path: Path to local file or URL to OpenAPI spec (JSON/YAML)
        
        Returns:
            OpenAPI specification content in JSON format
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_spec_path(spec_path)
            
            # Load the specification
            if self._is_url(spec_path):
                spec_data = self._read_from_url(spec_path)
            else:
                spec_data = self._read_from_file(spec_path)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self._metrics.total_execution_time += execution_time
            self._metrics.files_analyzed = 1
            
            # Always return as formatted JSON
            return json.dumps(spec_data, indent=2)
        
        except Exception as e:
            self._metrics.errors_encountered += 1
            return json.dumps({
                "error": f"Error reading OpenAPI spec: {str(e)}",
                "spec_source": spec_path,
                "metrics": asdict(self._metrics)
            }, indent=2)

    def _validate_spec_path(self, spec_path: str):
        """Validate the spec path input (supports both local files and URLs)."""
        if not spec_path or not isinstance(spec_path, str):
            raise ValidationError("spec_path must be a non-empty string")

    def _is_url(self, spec_path: str) -> bool:
        """Check if the spec_path is a URL"""
        parsed_url = urlparse(spec_path)
        return parsed_url.scheme in ('http', 'https')

    def _read_from_url(self, url: str) -> Dict[str, Any]:
        """Download OpenAPI spec from URL and return as dict"""
        try:
            print(f"Downloading OpenAPI spec from: {url}")
            response = requests.get(url, timeout=30, headers={'User-Agent': 'OpenAPI-Reader/1.0'})
            response.raise_for_status()
            
            # Try to parse as JSON first
            content_type = response.headers.get('content-type', '').lower()
            if 'json' in content_type or url.lower().endswith('.json'):
                try:
                    return response.json()
                except json.JSONDecodeError:
                    pass
            
            # Try to parse as YAML
            try:
                import yaml
                return yaml.safe_load(response.text)
            except ImportError:
                # If PyYAML is not installed, try JSON anyway
                try:
                    return response.json()
                except json.JSONDecodeError:
                    raise ToolError("PyYAML is required to parse YAML files. Please install it with: pip install PyYAML")
            except Exception as e:
                # Last resort - try JSON
                try:
                    return response.json()
                except json.JSONDecodeError:
                    raise ToolError(f"Failed to parse spec as JSON or YAML: {str(e)}")
                    
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Failed to download spec from {url}: {str(e)}")

    def _read_from_file(self, file_path: str) -> Dict[str, Any]:
        """Read OpenAPI spec from local file and return as dict"""
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Spec file does not exist: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try JSON first
            if file_path.lower().endswith('.json'):
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise ToolError(f"Invalid JSON in file {file_path}: {str(e)}")
            
            # For YAML files (.yaml, .yml) or others
            try:
                import yaml
                return yaml.safe_load(content)
            except ImportError:
                # If PyYAML not installed, try parsing as JSON anyway
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    raise ToolError("PyYAML is required to parse YAML files. Please install it with: pip install PyYAML")
            except Exception as e:
                # Last resort - try JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    raise ToolError(f"Failed to parse {file_path} as JSON or YAML: {str(e)}")
                    
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    if file_path.lower().endswith('.json'):
                        return json.loads(content)
                    else:
                        try:
                            import yaml
                            return yaml.safe_load(content)
                        except ImportError:
                            return json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            raise ToolError(f"Could not decode file {file_path} with any supported encoding")


class FastAPIAnalyzerTool(BaseTool):
    """Tool to analyze FastAPI applications and extract endpoint information from source code"""
    
    name: str = "FastAPI Analyzer"
    description: str = "Analyzes FastAPI source code from local folders to extract API endpoint specifications"

    def __init__(self):
        super().__init__()
        self._metrics = APIAnalysisMetrics()

    @handle_errors(ToolError)
    @retry_on_failure(max_attempts=3)
    @log_performance
    def _run(self, spec_path: str) -> str:
        """
        Analyze FastAPI source code and extract endpoint information
        
        Args:
            spec_path: Path to local folder containing FastAPI source code
        
        Returns:
            JSON string with extracted endpoint information
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_code_path(spec_path)
            
            # Analyze the code
            endpoints = self._analyze_fastapi_code(spec_path)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self._metrics.total_execution_time += execution_time
            
            result = {
                "framework": "fastapi",
                "source_path": spec_path,
                "endpoints": [endpoint.dict() for endpoint in endpoints],
                "total_endpoints": len(endpoints),
                "metrics": asdict(self._metrics),
                "timestamp": time.time()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            self._metrics.errors_encountered += 1
            return json.dumps({
                "framework": "fastapi",
                "error": f"Error analyzing FastAPI code: {str(e)}",
                "source_path": spec_path,
                "endpoints": [],
                "total_endpoints": 0,
                "metrics": asdict(self._metrics)
            }, indent=2)

    def _validate_code_path(self, spec_path: str):
        """Validate the code path input."""
        if not spec_path or not isinstance(spec_path, str):
            raise ValidationError("spec_path must be a non-empty string")
        
        path_obj = Path(spec_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Code path does not exist: {spec_path}")
        
        if not path_obj.is_dir():
            raise ValidationError(f"Code path must be a directory: {spec_path}")
    
    def _analyze_fastapi_code(self, code_path: str) -> List[APIEndpoint]:
        """Extract FastAPI endpoints from source code"""
        endpoints = []
        
        try:
            for root, dirs, files in os.walk(code_path):
                # Skip hidden directories and common non-source directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for file in files:
                    if file.endswith('.py') and not file.startswith('.'):
                        file_path = os.path.join(root, file)
                        try:
                            file_endpoints = self._parse_fastapi_file(file_path)
                            endpoints.extend(file_endpoints)
                            self._metrics.files_analyzed += 1
                        except Exception as e:
                            self._metrics.errors_encountered += 1
                            print(f"Error parsing file {file_path}: {str(e)}")
                            continue
            
            self._metrics.endpoints_found = len(endpoints)
            return endpoints
            
        except Exception as e:
            raise ToolError(f"Failed to analyze FastAPI code: {str(e)}")

    def _parse_fastapi_file(self, file_path: str) -> List[APIEndpoint]:
        """Parse a single FastAPI file for endpoints"""
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return endpoints
            
            tree = ast.parse(content)
            
            # Look for FastAPI router or app instances
            fastapi_instances = self._find_fastapi_instances(tree)
            
            # Extract endpoints from decorators
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    endpoint = self._extract_fastapi_endpoint(node, fastapi_instances)
                    if endpoint:
                        endpoints.append(endpoint)
        
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    tree = ast.parse(content)
                    
                    fastapi_instances = self._find_fastapi_instances(tree)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                            endpoint = self._extract_fastapi_endpoint(node, fastapi_instances)
                            if endpoint:
                                endpoints.append(endpoint)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ToolError(f"Could not decode file {file_path} with any supported encoding")
        except SyntaxError as e:
            print(f"Syntax error in file {file_path}: {str(e)}")
            return endpoints  # Return empty list for files with syntax errors
        except Exception as e:
            raise ToolError(f"Error parsing file {file_path}: {str(e)}")
        
        return endpoints

    def _find_fastapi_instances(self, tree: ast.AST) -> List[str]:
        """Find FastAPI app or router instance names"""
        instances = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check for app = FastAPI() or router = APIRouter()
                if isinstance(node.value, ast.Call):
                    if hasattr(node.value.func, 'id'):
                        if node.value.func.id in ['FastAPI', 'APIRouter']:
                            for target in node.targets:
                                if hasattr(target, 'id'):
                                    instances.append(target.id)
                    elif hasattr(node.value.func, 'attr'):
                        if node.value.func.attr in ['FastAPI', 'APIRouter']:
                            for target in node.targets:
                                if hasattr(target, 'id'):
                                    instances.append(target.id)
        
        # Common default names if none found
        if not instances:
            instances = ['app', 'router', 'api']
        
        return instances

    def _extract_fastapi_endpoint(self, func_node: ast.FunctionDef, fastapi_instances: List[str]) -> Optional[APIEndpoint]:
        """Extract endpoint information from FastAPI function"""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr'):
                # Check if decorator is on a FastAPI instance
                if hasattr(decorator.func.value, 'id') and decorator.func.value.id in fastapi_instances:
                    method = decorator.func.attr.lower()
                    if method in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                        path = self._extract_path_from_decorator(decorator)
                        
                        # Extract additional metadata
                        tags = self._extract_tags_from_decorator(decorator)
                        response_model = self._extract_response_model_from_decorator(decorator)
                        status_code = self._extract_status_code_from_decorator(decorator)
                        
                        # Get function docstring
                        docstring = ast.get_docstring(func_node)
                        
                        # Extract parameters from function signature
                        parameters = self._extract_parameters_from_function(func_node)
                        
                        endpoint = APIEndpoint(
                            path=path or "/unknown",
                            method=method.upper(),
                            handler_function=func_node.name,
                            description=docstring,
                            tags=tags if tags else None,
                            parameters=parameters if parameters else None
                        )
                        
                        # Add response information if available
                        if response_model or status_code:
                            endpoint.response_schemas = {
                                str(status_code or 200): {
                                    'description': 'Successful response',
                                    'model': response_model
                                }
                            }
                        
                        return endpoint
        
        return None

    def _extract_path_from_decorator(self, decorator: ast.Call) -> Optional[str]:
        """Extract path from FastAPI decorator"""
        if decorator.args:
            arg = decorator.args[0]
            if isinstance(arg, ast.Constant):
                return arg.value
            elif isinstance(arg, ast.Str):  # Python 3.7 compatibility
                return arg.s
        
        # Check for path keyword argument
        for keyword in decorator.keywords:
            if keyword.arg == 'path':
                if isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
                elif isinstance(keyword.value, ast.Str):
                    return keyword.value.s
        
        return None

    def _extract_tags_from_decorator(self, decorator: ast.Call) -> Optional[List[str]]:
        """Extract tags from FastAPI decorator"""
        for keyword in decorator.keywords:
            if keyword.arg == 'tags':
                if isinstance(keyword.value, ast.List):
                    tags = []
                    for elt in keyword.value.elts:
                        if isinstance(elt, ast.Constant):
                            tags.append(elt.value)
                        elif isinstance(elt, ast.Str):
                            tags.append(elt.s)
                    return tags if tags else None
        return None

    def _extract_response_model_from_decorator(self, decorator: ast.Call) -> Optional[str]:
        """Extract response_model from FastAPI decorator"""
        for keyword in decorator.keywords:
            if keyword.arg == 'response_model':
                if hasattr(keyword.value, 'id'):
                    return keyword.value.id
                elif hasattr(keyword.value, 'attr'):
                    return f"{keyword.value.value.id}.{keyword.value.attr}"
        return None

    def _extract_status_code_from_decorator(self, decorator: ast.Call) -> Optional[int]:
        """Extract status_code from FastAPI decorator"""
        for keyword in decorator.keywords:
            if keyword.arg == 'status_code':
                if isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
                elif isinstance(keyword.value, ast.Num):  # Python 3.7 compatibility
                    return keyword.value.n
        return None

    def _extract_parameters_from_function(self, func_node: ast.FunctionDef) -> Optional[List[Dict[str, Any]]]:
        """Extract parameters from function signature"""
        parameters = []
        
        # Skip 'self' parameter for methods
        args = func_node.args.args
        if args and args[0].arg in ['self', 'cls']:
            args = args[1:]
        
        for arg in args:
            param = {
                'name': arg.arg,
                'in': 'query',  # Default assumption
                'required': True
            }
            
            # Try to extract type annotation
            if arg.annotation:
                param['type'] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else 'string'
            
            parameters.append(param)
        
        return parameters if parameters else None


# Create tool instances for easy import
openapi_spec_reader = OpenAPISpecReaderTool
fastapi_analyzer = FastAPIAnalyzerTool