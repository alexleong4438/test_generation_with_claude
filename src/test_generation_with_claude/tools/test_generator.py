"""
Enhanced Test Generation Tools for Pytest and API Testing without External Utils
"""

import json
import os
import subprocess
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from functools import wraps
from contextlib import contextmanager
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# Setup basic logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Custom exceptions
class ToolDataError(Exception):
    """Exception for tool data related errors"""
    pass


class ToolConfigurationError(Exception):
    """Exception for tool configuration errors"""
    pass


# Simple retry decorator
def retry_file_operation(max_attempts=3, delay=1):
    """Simple retry decorator for file operations"""
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
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
            raise last_exception
        return wrapper
    return decorator


# Simple performance logger context manager
@contextmanager
def performance_logger(operation_name):
    """Simple context manager for logging performance"""
    start_time = time.time()
    logger.info(f"Starting {operation_name}")
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name} in {duration:.2f} seconds")


# Simple error handling context manager
@contextmanager
def error_handling_context(context_info):
    """Simple context manager for error handling"""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {context_info.get('component', 'unknown')} - "
                    f"{context_info.get('operation', 'unknown')}: {e}")
        raise


class TestCase(BaseModel):
    """Model for test case information"""
    name: str
    description: str
    endpoint: str
    method: str
    test_type: str  # happy_path, error_case, edge_case
    priority: str
    setup_data: Optional[Dict[str, Any]] = None
    expected_response: Optional[Dict[str, Any]] = None


class EnhancedPytestGeneratorTool(BaseTool):
    """Enhanced tool to generate pytest code for API testing with error handling"""
    
    name: str = "Enhanced Pytest Generator"
    description: str = "Generates high-quality pytest code for API testing with enhanced reliability"
    templates_dir: str = Field(
        default="templates",
        description="Directory containing test templates for pytest generation"
    )
    output_cache: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cache for generated test files to avoid redundant generation"
    )
    generation_stats: Dict[str, Any] = Field(
        default_factory=lambda: {
            'total_generated': 0,
            'successful_generations': 0,
            'failed_generations': 0
        },
        description="Statistics for test generation operations"
    )

    def __init__(self, templates_dir: str = None, **kwargs):
        super().__init__(**kwargs)
        self.templates_dir = templates_dir or "templates"
        self.output_cache = {}
        self.generation_stats = {
            'total_generated': 0,
            'successful_generations': 0,
            'failed_generations': 0
        }

    def _run(self, test_scenarios: str, output_path: str = "tests") -> str:
        """Enhanced run method with comprehensive error handling"""
        
        context_info = {
            'component': 'pytest_generator',
            'operation': 'generate_tests',
            'output_path': output_path
        }
        
        with error_handling_context(context_info):
            with performance_logger("pytest_generation"):
                try:
                    # Validate inputs
                    if not test_scenarios:
                        raise ToolDataError("Test scenarios cannot be empty")
                    
                    # Parse scenarios
                    try:
                        print(f"Parsing test scenarios: {test_scenarios}")
                        scenarios = json.loads(test_scenarios)
                    except json.JSONDecodeError as e:
                        raise ToolDataError(f"Invalid JSON in test scenarios: {e}")
                    
                    # Generate test files
                    generated_files = self._generate_test_files_with_retry(scenarios, output_path)
                    
                    # Update statistics
                    self.generation_stats['total_generated'] += 1
                    self.generation_stats['successful_generations'] += 1
                    
                    result = {
                        "status": "success",
                        "generated_files": generated_files,
                        "total_tests": sum(len(file_info["tests"]) for file_info in generated_files.values()),
                        "generation_timestamp": time.time(),
                        "output_path": output_path
                    }
                    
                    return json.dumps(result, indent=2)
                
                except Exception as e:
                    self.generation_stats['failed_generations'] += 1
                    logger.error(f"Test generation failed: {e}")
                    raise ToolDataError(f"Failed to generate pytest code: {e}")

    @retry_file_operation(max_attempts=3, delay=1)
    def _generate_test_files_with_retry(self, scenarios: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Generate test files with retry logic"""
        return self._generate_test_files(scenarios, output_path)

    def _generate_test_files(self, scenarios: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Generate test files from scenarios with enhanced validation"""
        generated_files = {}
        
        try:
            # Validate scenarios structure
            if not isinstance(scenarios, dict):
                raise ToolDataError("Scenarios must be a dictionary")
            
            test_scenarios = scenarios.get("test_scenarios", [])
            if not isinstance(test_scenarios, list):
                raise ToolDataError("test_scenarios must be a list")
            
            # Group scenarios by endpoint
            endpoints = {}
            for scenario in test_scenarios:
                if not isinstance(scenario, dict):
                    logger.warning("Skipping invalid scenario (not a dictionary)")
                    continue
                
                endpoint = scenario.get("endpoint", "unknown")
                if endpoint not in endpoints:
                    endpoints[endpoint] = []
                endpoints[endpoint].append(scenario)
            
            # Generate test file for each endpoint
            for endpoint, endpoint_scenarios in endpoints.items():
                try:
                    file_name = f"test_{self._sanitize_filename(endpoint)}.py"
                    file_path = os.path.join(output_path, "api", file_name)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Generate test content
                    test_content = self._generate_test_file_content(endpoint, endpoint_scenarios)
                    
                    # Validate generated content
                    self._validate_generated_content(test_content, file_path)
                    
                    # Write file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(test_content)
                    
                    # Verify file was written correctly
                    if not os.path.exists(file_path):
                        raise ToolDataError(f"Failed to create test file: {file_path}")
                    
                    generated_files[file_name] = {
                        "path": file_path,
                        "endpoint": endpoint,
                        "tests": [s.get("name", f"test_{i}") for i, s in enumerate(endpoint_scenarios)],
                        "size_bytes": os.path.getsize(file_path)
                    }
                    
                    logger.info(f"Generated test file: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate test file for endpoint {endpoint}: {e}")
                    # Continue with other endpoints
                    continue
            
            return generated_files
            
        except Exception as e:
            raise ToolDataError(f"Test file generation failed: {e}")

    def _validate_generated_content(self, content: str, file_path: str):
        """Validate generated test content"""
        if not content or len(content.strip()) < 100:
            raise ToolDataError(f"Generated content too short for {file_path}")
        
        # Check for basic Python syntax
        required_patterns = ['import', 'def test_', 'class Test']
        if not any(pattern in content for pattern in required_patterns):
            raise ToolDataError(f"Generated content doesn't appear to be valid test code for {file_path}")
        
        # Basic syntax check (optional, requires ast module)
        try:
            import ast
            ast.parse(content)
        except SyntaxError as e:
            raise ToolDataError(f"Generated content has syntax errors: {e}")

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize string for use as filename"""
        import re
        # Remove non-alphanumeric characters except underscore and hyphen
        sanitized = re.sub(r'[^\w\-]', '_', name.lower())
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')

    def _to_pascal_case(self, text: str) -> str:
        """Convert text to PascalCase"""
        import re
        words = re.findall(r'[a-zA-Z0-9]+', text)
        return ''.join(word.capitalize() for word in words)

    def _generate_test_file_content(self, endpoint: str, scenarios: List[Dict[str, Any]]) -> str:
        """Generate content for a test file with enhanced structure"""
        class_name = f"Test{self._to_pascal_case(endpoint)}"
        
        content = f'''"""
API tests for {endpoint}
Generated automatically from requirements with enhanced error handling
"""

import pytest
import httpx
import json
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch
import asyncio


class {class_name}:
    """Test class for {endpoint} endpoint"""
    
    @pytest.fixture
    def client(self):
        """HTTP client fixture"""
        return httpx.AsyncClient()
    
    @pytest.fixture
    def base_url(self):
        """Base URL fixture"""
        return "https://api.example.com"
    
    @pytest.fixture
    def auth_headers(self):
        """Authentication headers fixture"""
        return {{"Authorization": "Bearer test-token"}}
'''

        # Generate test methods for each scenario
        for i, scenario in enumerate(scenarios):
            content += self._generate_test_method(scenario, i)
        
        # Add utility methods
        content += '''
    def _assert_response_structure(self, response: httpx.Response, expected_fields: list):
        """Assert response has expected structure"""
        assert response.status_code < 400, f"Request failed with status {response.status_code}"
        
        try:
            data = response.json()
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")
        
        for field in expected_fields:
            assert field in data, f"Expected field '{field}' not found in response"
    
    def _assert_error_response(self, response: httpx.Response, expected_status: int):
        """Assert error response format"""
        assert response.status_code == expected_status, f"Expected status {expected_status}, got {response.status_code}"
        
        try:
            error_data = response.json()
            assert "error" in error_data or "message" in error_data, "Error response should contain error information"
        except json.JSONDecodeError:
            # Some APIs return plain text errors
            assert len(response.text) > 0, "Error response should not be empty"
'''
        
        return content

    def _generate_test_method(self, scenario: Dict[str, Any], index: int) -> str:
        """Generate a test method for a scenario"""
        scenario_name = scenario.get("name", f"test_scenario_{index}")
        method_name = self._sanitize_filename(scenario_name)
        endpoint = scenario.get("endpoint", "")
        test_type = scenario.get("type", "happy_path")
        description = scenario.get("description", f"Test scenario {index}")
        
        # Extract HTTP method and path from endpoint
        if " " in endpoint:
            http_method, path = endpoint.split(" ", 1)
        else:
            http_method = "GET"
            path = endpoint
        
        content = f'''
    @pytest.mark.asyncio
    async def test_{method_name}(self, client, base_url, auth_headers):
        """
        {description}
        
        Test type: {test_type}
        Endpoint: {endpoint}
        """
        url = f"{{base_url}}{path}"
        
'''
        
        if test_type == "happy_path":
            content += self._generate_happy_path_test(http_method, scenario)
        elif test_type == "error_handling":
            content += self._generate_error_test(http_method, scenario)
        else:
            content += self._generate_generic_test(http_method, scenario)
        
        return content

    def _generate_happy_path_test(self, method: str, scenario: Dict[str, Any]) -> str:
        """Generate happy path test code"""
        method_lower = method.lower()
        
        if method_lower == "get":
            return '''        # Test successful GET request
        response = await client.get(url, headers=auth_headers)
        
        # Assert successful response
        assert response.status_code == 200
        self._assert_response_structure(response, ["data"])
        
        # Verify response data
        data = response.json()
        assert isinstance(data, dict)
'''
        elif method_lower == "post":
            return '''        # Test successful POST request
        test_data = {"name": "test", "value": "example"}
        
        response = await client.post(url, json=test_data, headers=auth_headers)
        
        # Assert successful creation
        assert response.status_code in [200, 201]
        self._assert_response_structure(response, ["id"])
        
        # Verify created resource
        data = response.json()
        assert "id" in data
'''
        elif method_lower == "put":
            return '''        # Test successful PUT request
        test_data = {"name": "updated", "value": "example"}
        
        response = await client.put(url, json=test_data, headers=auth_headers)
        
        # Assert successful update
        assert response.status_code in [200, 204]
        
        if response.status_code == 200:
            self._assert_response_structure(response, ["id"])
'''
        elif method_lower == "delete":
            return '''        # Test successful DELETE request
        response = await client.delete(url, headers=auth_headers)
        
        # Assert successful deletion
        assert response.status_code in [200, 204]
'''
        else:
            return '''        # Test successful request
        response = await client.request(method.upper(), url, headers=auth_headers)
        
        # Assert successful response
        assert response.status_code < 400
'''

    def _generate_error_test(self, method: str, scenario: Dict[str, Any]) -> str:
        """Generate error handling test code"""
        method_lower = method.lower()
        
        return f'''        # Test error handling for {method.upper()} request
        
        # Test with invalid data
        if "{method_lower}" in ["post", "put", "patch"]:
            invalid_data = {{"invalid": "data"}}
            response = await client.{method_lower}(url, json=invalid_data, headers=auth_headers)
            self._assert_error_response(response, 400)
        
        # Test without authentication
        response = await client.{method_lower}(url)
        self._assert_error_response(response, 401)
        
        # Test with invalid endpoint
        invalid_url = url + "/nonexistent"
        response = await client.{method_lower}(invalid_url, headers=auth_headers)
        self._assert_error_response(response, 404)
'''

    def _generate_generic_test(self, method: str, scenario: Dict[str, Any]) -> str:
        """Generate generic test code"""
        method_lower = method.lower()
        
        return f'''        # Generic test for {method.upper()} request
        response = await client.{method_lower}(url, headers=auth_headers)
        
        # Basic assertions
        assert response.status_code < 500, "Server should not return 5xx errors"
        
        # Verify response format
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                response.json()
            except json.JSONDecodeError:
                pytest.fail("Response claims to be JSON but is not valid JSON")
'''

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.generation_stats.copy()

    def clear_cache(self):
        """Clear output cache"""
        self.output_cache.clear()


class HTTPXTemplateToolInput(BaseModel):
    """Input model for HTTPX test generation"""
    endpoint_spec: str = Field(..., description="JSON string specifying the endpoint and method for test generation")

class EnhancedHTTPXTemplateTool(BaseTool):
    """Enhanced tool for generating HTTPX test templates"""
    
    name: str = "Enhanced HTTPX Template Generator"
    description: str = "Generates HTTPX-based test templates with enhanced error handling"
    args_schema: Type[BaseModel] = HTTPXTemplateToolInput

    def _run(self, endpoint_spec: str) -> str:
        """Generate HTTPX template from endpoint specification"""
        
        context_info = {
            'component': 'httpx_template',
            'operation': 'generate_template'
        }
        
        with error_handling_context(context_info):
            with performance_logger("httpx_template_generation"):
                try:
                    spec = json.loads(endpoint_spec)
                    template = self._generate_httpx_template(spec)
                    
                    return json.dumps({
                        "status": "success",
                        "template": template
                    }, indent=2)
                    
                except json.JSONDecodeError as e:
                    raise ToolDataError(f"Invalid endpoint specification JSON: {e}")
                except Exception as e:
                    raise ToolDataError(f"Template generation failed: {e}")

    def _generate_httpx_template(self, spec: Dict[str, Any]) -> str:
        """Generate HTTPX template code"""
        endpoint = spec.get("endpoint", "/api/example")
        method = spec.get("method", "GET").upper()
        
        template = f'''
async def test_{method.lower()}_{endpoint.replace("/", "_").strip("_")}():
    """Test {method} {endpoint}"""
    async with httpx.AsyncClient() as client:
        response = await client.{method.lower()}(
            "https://api.example.com{endpoint}",
            headers={{"Authorization": "Bearer token"}}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
'''
        return template

class XrayAPIFixtureGeneratorToolInput(BaseModel):
    """Input model for Xray API fixture generation"""
    fixture_spec: str = Field(...,description="JSON string specifying the fixture requirements")

class EnhancedFixtureGeneratorTool(BaseTool):
    """Enhanced tool for generating pytest fixtures"""
    
    name: str = "Enhanced Fixture Generator"
    description: str = "Generates pytest fixtures with enhanced patterns"
    args_schema: Type[BaseModel] = XrayAPIFixtureGeneratorToolInput

    def _run(self, fixture_spec: str) -> str:
        """Generate pytest fixtures from specification"""
        
        context_info = {
            'component': 'fixture_generator',
            'operation': 'generate_fixtures'
        }
        
        with error_handling_context(context_info):
            try:
                spec = json.loads(fixture_spec)
                fixtures = self._generate_fixtures(spec)
                
                return json.dumps({
                    "status": "success",
                    "fixtures": fixtures
                }, indent=2)
                
            except json.JSONDecodeError as e:
                raise ToolDataError(f"Invalid fixture specification JSON: {e}")
            except Exception as e:
                raise ToolDataError(f"Fixture generation failed: {e}")

    def _generate_fixtures(self, spec: Dict[str, Any]) -> str:
        """Generate fixture code"""
        fixture_type = spec.get("type", "client")
        
        if fixture_type == "client":
            return '''
@pytest.fixture
async def http_client():
    """HTTP client fixture with proper cleanup"""
    async with httpx.AsyncClient() as client:
        yield client

@pytest.fixture
def auth_headers():
    """Authentication headers fixture"""
    return {"Authorization": "Bearer test-token"}
'''
        elif fixture_type == "database":
            return '''
@pytest.fixture
async def db_session():
    """Database session fixture"""
    # Database setup code here
    session = create_test_session()
    try:
        yield session
    finally:
        session.close()
'''
        else:
            return '''
@pytest.fixture
def test_data():
    """Test data fixture"""
    return {"test": "data"}
'''


# Backward compatibility aliases
PytestGeneratorTool = EnhancedPytestGeneratorTool
HTTPXTemplateTool = EnhancedHTTPXTemplateTool
FixtureGeneratorTool = EnhancedFixtureGeneratorTool