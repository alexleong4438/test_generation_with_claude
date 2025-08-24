"""
Enhanced Xray API tool with comprehensive test case management, bulk operations, and test execution support.
"""
import os
import base64
import requests
import time
import json
from typing import Type, Optional, Dict, Any, List, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import logging
from functools import wraps

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Simple retry decorator
def retry_on_failure(max_attempts=3, delay=1.0):
    """Simple retry decorator for functions."""
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
                        sleep_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed.")
            raise last_exception
        return wrapper
    return decorator


# Simple performance logger
class PerformanceLogger:
    """Simple context manager for performance logging."""
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(f"{self.operation_name} completed in {duration:.3f} seconds")


class TestExecutionStatus(str, Enum):
    """Xray test execution statuses"""
    PASS = "PASS"
    FAIL = "FAIL"
    TODO = "TODO"
    EXECUTING = "EXECUTING"
    BLOCKED = "BLOCKED"

class XrayOperation(str, Enum):
    """Supported Xray operations"""
    GET_TEST_CASE = "get_test_case"
    CREATE_TEST_CASE = "create_test_case"
    UPDATE_TEST_CASE = "update_test_case"
    CREATE_TEST_EXECUTION = "create_test_execution"
    UPDATE_TEST_EXECUTION = "update_test_execution"
    GET_TEST_SET = "get_test_set"
    BULK_IMPORT = "bulk_import"
    GET_TEST_PLAN = "get_test_plan"


class XrayAPIInput(BaseModel):
    """Input schema for Xray API operations"""
    operation: XrayOperation = Field(description="The operation to perform")
    test_case_key: Optional[str] = Field(None, description="Test case key for single operations")
    test_case_keys: Optional[List[str]] = Field(None, description="List of test case keys for bulk operations")
    include_steps: bool = Field(default=True, description="Whether to include test steps")
    include_attachments: bool = Field(default=False, description="Whether to include attachments")
    include_requirements: bool = Field(default=False, description="Whether to include linked requirements")
    include_executions: bool = Field(default=False, description="Whether to include test executions")
    test_data: Optional[Dict[str, Any]] = Field(None, description="Data for create/update operations")
    project_key: Optional[str] = Field(None, description="Project key for creating new test cases")
    
    @field_validator('operation', mode='before')
    @classmethod
    def validate_operation(cls, v):
        """Validate operation field - accept both enum constants and string values"""
        if isinstance(v, str):
            # Handle cases where CrewAI passes the enum name instead of value
            if v in ['GET_TEST_CASE', 'CREATE_TEST_CASE', 'UPDATE_TEST_CASE', 
                     'CREATE_TEST_EXECUTION', 'UPDATE_TEST_EXECUTION', 
                     'GET_TEST_SET', 'BULK_IMPORT', 'GET_TEST_PLAN']:
                # Convert enum constant name to enum value
                enum_mapping = {
                    'GET_TEST_CASE': 'get_test_case',
                    'CREATE_TEST_CASE': 'create_test_case', 
                    'UPDATE_TEST_CASE': 'update_test_case',
                    'CREATE_TEST_EXECUTION': 'create_test_execution',
                    'UPDATE_TEST_EXECUTION': 'update_test_execution',
                    'GET_TEST_SET': 'get_test_set',
                    'BULK_IMPORT': 'bulk_import',
                    'GET_TEST_PLAN': 'get_test_plan'
                }
                return enum_mapping.get(v, v)
            # If it's already the correct lowercase value, return as-is
            return v        # If it's already an enum instance, return as-is
        return v
    
    def model_post_init(self, __context):
        """Validate the complete model after all fields are processed."""
        # Check if operation requires keys
        operations_requiring_keys = [
            XrayOperation.GET_TEST_CASE, 
            XrayOperation.UPDATE_TEST_CASE
        ]
        
        if self.operation in operations_requiring_keys:
            # Require either test_case_key or test_case_keys for these operations
            if not self.test_case_key and not self.test_case_keys:
                raise ValueError("test_case_key or test_case_keys required for this operation")
        
        # Validate project_key is provided for create operations
        if self.operation == XrayOperation.CREATE_TEST_CASE and not self.project_key:
            raise ValueError("project_key is required for creating new test cases")


class TestStep(BaseModel):
    """Model for test step data"""
    action: str
    data: Optional[str] = None
    expected_result: Optional[str] = None
    attachments: Optional[List[str]] = None


class TestCase(BaseModel):
    """Model for test case data"""
    key: Optional[str] = None
    summary: str
    description: Optional[str] = None
    test_type: str = "Manual"
    priority: str = "Medium"
    labels: List[str] = []
    steps: List[TestStep] = []
    preconditions: Optional[str] = None
    objective: Optional[str] = None
    requirements: List[str] = []
    components: List[str] = []
    fix_versions: List[str] = []


class XrayConnectionError(Exception):
    """Custom exception for Xray connection issues."""
    pass


class XrayAuthenticationError(Exception):
    """Custom exception for Xray authentication issues."""
    pass


class XrayDataError(Exception):
    """Custom exception for Xray data issues."""
    pass


class EnhancedXrayAPITool(BaseTool):
    """Enhanced Xray API tool with comprehensive test management capabilities."""
    
    name: str = "enhanced_xray_api_tool"
    description: str = """
    Comprehensive Xray API tool that supports:
    - Retrieving test cases with all details
    - Creating and updating test cases
    - Managing test executions
    - Bulk operations
    - Test sets and test plans
    - Requirements traceability
    """
    args_schema: Type[BaseModel] = XrayAPIInput
    
    # Add these fields to avoid Pydantic validation errors
    base_url: str = Field(default="", description="Jira base URL")
    user_email: str = Field(default="", description="Jira user email")
    api_token: str = Field(default="", description="Jira API token")
    custom_fields: Dict[str, str] = Field(default_factory=dict, description="Custom field mappings")
    request_timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_workers: int = Field(default=5, description="Maximum worker threads")
    session: Optional[requests.Session] = Field(default=None, description="HTTP session")
    cache: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Internal cache")
    cache_ttl: Dict[str, int] = Field(default_factory=dict, description="Cache TTL settings")
    
    class Config:
        arbitrary_types_allowed = True  # Allow requests.Session and other complex types

    def __init__(self, **kwargs):
        # Initialize fields first before calling super().__init__()
        base_url = os.getenv("JIRA_BASE_URL", "").rstrip('/')
        user_email = os.getenv("JIRA_EMAIL", "")
        api_token = os.getenv("JIRA_API_TOKEN", "")
        
        # Xray-specific custom fields (configurable via environment)
        custom_fields = {
            'steps': os.getenv("XRAY_STEPS_FIELD", "customfield_10000"),
            'test_type': os.getenv("XRAY_TEST_TYPE_FIELD", "customfield_10001"),
            'preconditions': os.getenv("XRAY_PRECONDITIONS_FIELD", "customfield_10002"),
            'test_repository_path': os.getenv("XRAY_TEST_REPO_PATH_FIELD", "customfield_10003"),
            'test_execution_status': os.getenv("XRAY_EXECUTION_STATUS_FIELD", "customfield_10004"),
            'objective': os.getenv("XRAY_OBJECTIVE_FIELD", "customfield_10005")
        }
        
        request_timeout = float(os.getenv("JIRA_TIMEOUT", "30"))
        max_workers = int(os.getenv("XRAY_MAX_WORKERS", "5"))
        
        # Enhanced cache with TTL per key type
        cache_ttl = {
            'test_case': 300,  # 5 minutes
            'test_set': 600,   # 10 minutes
            'test_plan': 600,  # 10 minutes
            'execution': 60    # 1 minute
        }
        
        # Set defaults in kwargs if not provided
        kwargs.setdefault('base_url', base_url)
        kwargs.setdefault('user_email', user_email)
        kwargs.setdefault('api_token', api_token)
        kwargs.setdefault('custom_fields', custom_fields)
        kwargs.setdefault('request_timeout', request_timeout)
        kwargs.setdefault('max_workers', max_workers)
        kwargs.setdefault('cache', {})
        kwargs.setdefault('cache_ttl', cache_ttl)
        
        super().__init__(**kwargs)
        
        # Session for connection pooling
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Setup requests session with authentication and headers."""
        if not all([self.base_url, self.user_email, self.api_token]):
            raise XrayAuthenticationError("Missing required Xray configuration")
        
        # Setup authentication
        auth_string = f"{self.user_email}:{self.api_token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        self.session.headers.update({
            "Authorization": f"Basic {encoded_auth}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-XrayAgent/2.0"
        })
          # Enhanced connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # We handle retries ourselves
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
    
    def _run(self, operation: XrayOperation, test_case_key: Optional[str] = None, 
             test_case_keys: Optional[List[str]] = None, include_steps: bool = True,
             include_attachments: bool = False, include_requirements: bool = False,
             include_executions: bool = False, test_data: Optional[Dict[str, Any]] = None,
             project_key: Optional[str] = None, **kwargs) -> str:
        """Main entry point for all Xray operations."""
        
        # Create and validate input using the schema
        try:
            validated_input = XrayAPIInput(
                operation=operation,
                test_case_key=test_case_key,
                test_case_keys=test_case_keys,
                include_steps=include_steps,
                include_attachments=include_attachments,
                include_requirements=include_requirements,
                include_executions=include_executions,
                test_data=test_data,
                project_key=project_key
            )
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise
        
        context = {
            'component': 'xray_api',
            'operation': operation.value,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with PerformanceLogger(f"xray_{operation.value}"):
            try:
                # Route to appropriate handler using validated input
                if operation == XrayOperation.GET_TEST_CASE:
                    return self._handle_get_test_case(
                        test_case_key=validated_input.test_case_key,
                        test_case_keys=validated_input.test_case_keys,
                        include_steps=validated_input.include_steps,
                        include_attachments=validated_input.include_attachments,
                        include_requirements=validated_input.include_requirements,
                        include_executions=validated_input.include_executions
                    )
                elif operation == XrayOperation.CREATE_TEST_CASE:
                    return self._handle_create_test_case(
                        project_key=validated_input.project_key,
                        test_data=validated_input.test_data
                    )
                elif operation == XrayOperation.UPDATE_TEST_CASE:
                    return self._handle_update_test_case(
                        test_case_key=validated_input.test_case_key,
                        test_data=validated_input.test_data
                    )
                elif operation == XrayOperation.CREATE_TEST_EXECUTION:
                    return self._handle_create_test_execution(
                        test_data=validated_input.test_data
                    )
                elif operation == XrayOperation.UPDATE_TEST_EXECUTION:
                    return self._handle_update_test_execution(
                        test_case_key=validated_input.test_case_key,
                        test_data=validated_input.test_data
                    )
                elif operation == XrayOperation.GET_TEST_SET:
                    return self._handle_get_test_set(
                        test_case_key=validated_input.test_case_key
                    )
                elif operation == XrayOperation.BULK_IMPORT:
                    return self._handle_bulk_import(
                        test_data=validated_input.test_data
                    )
                elif operation == XrayOperation.GET_TEST_PLAN:
                    return self._handle_get_test_plan(
                        test_case_key=validated_input.test_case_key
                    )
                else:
                    raise XrayDataError(f"Unsupported operation: {operation}")
                    
            except Exception as e:
                logger.error(f"Xray operation {operation} failed: {e}")
                raise
    
    def _handle_get_test_case(self, test_case_key: str = None, test_case_keys: List[str] = None, 
                            include_steps: bool = True, include_attachments: bool = False,
                            include_requirements: bool = False, include_executions: bool = False, 
                            **kwargs) -> str:
        """Handle getting test case(s)."""
        
        if test_case_keys:
            # Bulk fetch
            return self._bulk_get_test_cases(
                test_case_keys, include_steps, include_attachments, 
                include_requirements, include_executions
            )
        elif test_case_key:
            # Single fetch
            return self._get_single_test_case(
                test_case_key, include_steps, include_attachments,
                include_requirements, include_executions
            )
        else:
            raise XrayDataError("No test case key(s) provided")
    
    @retry_on_failure(max_attempts=3)
    def _get_single_test_case(self, test_case_key: str, include_steps: bool,
                            include_attachments: bool, include_requirements: bool,
                            include_executions: bool) -> str:
        """Fetch a single test case with enhanced details."""
        
        # Check cache
        cache_key = f"tc_{test_case_key}_{include_steps}_{include_attachments}_{include_requirements}_{include_executions}"
        cached = self._get_from_cache(cache_key, 'test_case')
        if cached:
            return json.dumps(cached, indent=2)
        
        try:
            # Build fields list
            fields = ["summary", "description", "status", "priority", "labels", 
                     "components", "fixVersions", "reporter", "assignee", "created", "updated"]
            
            if include_steps:
                fields.extend([self.custom_fields['steps'], self.custom_fields['preconditions']])
            
            if include_attachments:
                fields.append("attachment")
            
            if include_requirements:
                fields.append("issuelinks")
            
            # Add Xray custom fields
            fields.extend([
                self.custom_fields['test_type'],
                self.custom_fields['objective'],
                self.custom_fields['test_repository_path']
            ])
            
            url = f"{self.base_url}/rest/api/3/issue/{test_case_key}"
            params = {
                "fields": ",".join(fields),
                "expand": "renderedFields,names,schema"
            }
            
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            self._handle_response_errors(response, test_case_key)
            
            data = response.json()
            
            # Get executions if requested
            executions = []
            if include_executions:
                executions = self._get_test_executions(test_case_key)
            
            # Process and format the data
            result = self._format_test_case_data(data, include_steps, include_attachments,
                                               include_requirements, executions)
            
            # Cache the result
            self._add_to_cache(cache_key, result, 'test_case')
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to fetch test case {test_case_key}: {e}")
            raise
    
    def _bulk_get_test_cases(self, test_case_keys: List[str], include_steps: bool,
                            include_attachments: bool, include_requirements: bool,
                            include_executions: bool) -> str:
        """Fetch multiple test cases in parallel."""
        
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(
                    self._get_single_test_case, key, include_steps,
                    include_attachments, include_requirements, include_executions
                ): key for key in test_case_keys
            }
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    results.append(json.loads(result))
                except Exception as e:
                    logger.error(f"Failed to fetch test case {key}: {e}")
                    errors.append({'key': key, 'error': str(e)})
        
        return json.dumps({
            'test_cases': results,
            'errors': errors,
            'summary': {
                'total_requested': len(test_case_keys),
                'successful': len(results),
                'failed': len(errors)
            }
        }, indent=2)
    
    def _handle_create_test_case(self, project_key: str, test_data: Dict[str, Any], **kwargs) -> str:
        """Create a new test case in Xray."""
        
        if not project_key:
            raise XrayDataError("Project key is required for creating test cases")
        
        if not test_data:
            raise XrayDataError("Test data is required for creating test cases")
        
        try:
            # Validate and prepare test case data
            test_case = TestCase(**test_data)
            
            # Build Jira issue data
            issue_data = {
                "fields": {
                    "project": {"key": project_key},
                    "issuetype": {"name": "Test"},
                    "summary": test_case.summary,
                    "description": self._format_description(test_case.description),
                    "priority": {"name": test_case.priority},
                    "labels": test_case.labels,
                    self.custom_fields['test_type']: test_case.test_type
                }
            }
            
            # Add optional fields
            if test_case.components:
                issue_data["fields"]["components"] = [{"name": c} for c in test_case.components]
            
            if test_case.fix_versions:
                issue_data["fields"]["fixVersions"] = [{"name": v} for v in test_case.fix_versions]
            
            if test_case.preconditions:
                issue_data["fields"][self.custom_fields['preconditions']] = test_case.preconditions
            
            if test_case.objective:
                issue_data["fields"][self.custom_fields['objective']] = test_case.objective
            
            # Format test steps for Xray
            if test_case.steps:
                issue_data["fields"][self.custom_fields['steps']] = self._format_test_steps(test_case.steps)
            
            # Create the issue
            url = f"{self.base_url}/rest/api/3/issue"
            response = self.session.post(url, json=issue_data, timeout=self.request_timeout)
            self._handle_response_errors(response, "create test case")
            
            created_issue = response.json()
            
            # Link requirements if provided
            if test_case.requirements:
                self._link_requirements(created_issue['key'], test_case.requirements)
            
            return json.dumps({
                'success': True,
                'test_case_key': created_issue['key'],
                'test_case_id': created_issue['id'],
                'self': created_issue['self'],
                'message': f"Test case {created_issue['key']} created successfully"
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to create test case: {e}")
            raise
    
    def _handle_update_test_case(self, test_case_key: str, test_data: Dict[str, Any], **kwargs) -> str:
        """Update an existing test case."""
        
        if not test_case_key:
            raise XrayDataError("Test case key is required for updating")
        
        if not test_data:
            raise XrayDataError("Test data is required for updating")
        
        try:
            # Build update data
            update_fields = {}
            
            # Map simple fields
            field_mapping = {
                'summary': 'summary',
                'description': 'description',
                'priority': lambda x: {"name": x},
                'labels': 'labels',
                'test_type': self.custom_fields['test_type'],
                'preconditions': self.custom_fields['preconditions'],
                'objective': self.custom_fields['objective']
            }
            
            for field, value in test_data.items():
                if field in field_mapping and value is not None:
                    if callable(field_mapping[field]):
                        update_fields[field] = field_mapping[field](value)
                    else:
                        update_fields[field_mapping[field]] = value
            
            # Handle complex fields
            if 'steps' in test_data:
                update_fields[self.custom_fields['steps']] = self._format_test_steps(test_data['steps'])
            
            if 'components' in test_data:
                update_fields['components'] = [{"name": c} for c in test_data['components']]
            
            if 'fix_versions' in test_data:
                update_fields['fixVersions'] = [{"name": v} for v in test_data['fix_versions']]
            
            # Update the issue
            url = f"{self.base_url}/rest/api/3/issue/{test_case_key}"
            response = self.session.put(url, json={"fields": update_fields}, timeout=self.request_timeout)
            self._handle_response_errors(response, f"update test case {test_case_key}")
            
            return json.dumps({
                'success': True,
                'test_case_key': test_case_key,
                'updated_fields': list(update_fields.keys()),
                'message': f"Test case {test_case_key} updated successfully"
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to update test case {test_case_key}: {e}")
            raise
    
    def _handle_create_test_execution(self, test_case_keys: List[str], 
                                    execution_data: Dict[str, Any], **kwargs) -> str:
        """Create a test execution for given test cases."""
        
        if not test_case_keys:
            raise XrayDataError("Test case keys are required for creating execution")
        
        try:
            project_key = execution_data.get('project_key')
            if not project_key:
                # Get project from first test case
                first_tc = self._get_issue_basic_info(test_case_keys[0])
                project_key = first_tc['fields']['project']['key']
            
            # Create test execution issue
            execution_issue = {
                "fields": {
                    "project": {"key": project_key},
                    "issuetype": {"name": "Test Execution"},
                    "summary": execution_data.get('summary', f"Test Execution - {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
                    "description": execution_data.get('description', 'Automated test execution')
                }
            }
            
            # Add optional fields
            if 'assignee' in execution_data:
                execution_issue['fields']['assignee'] = {"name": execution_data['assignee']}
            
            if 'fix_version' in execution_data:
                execution_issue['fields']['fixVersions'] = [{"name": execution_data['fix_version']}]
            
            # Create the execution
            url = f"{self.base_url}/rest/api/3/issue"
            response = self.session.post(url, json=execution_issue, timeout=self.request_timeout)
            self._handle_response_errors(response, "create test execution")
            
            execution = response.json()
            
            # Add tests to execution
            self._add_tests_to_execution(execution['key'], test_case_keys)
            
            return json.dumps({
                'success': True,
                'execution_key': execution['key'],
                'execution_id': execution['id'],
                'test_cases': test_case_keys,
                'message': f"Test execution {execution['key']} created with {len(test_case_keys)} test cases"
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to create test execution: {e}")
            raise
    
    def _handle_update_test_execution(self, execution_key: str, test_results: Dict[str, Any], **kwargs) -> str:
        """Update test execution results."""
        
        if not execution_key:
            raise XrayDataError("Execution key is required for updating results")
        
        try:
            results = []
            
            for test_key, result_data in test_results.items():
                test_run = {
                    "testKey": test_key,
                    "status": result_data.get('status', TestExecutionStatus.TODO.value),
                    "comment": result_data.get('comment', ''),
                    "executedBy": result_data.get('executed_by', self.user_email)
                }
                
                # Add evidence if provided
                if 'evidence' in result_data:
                    test_run['evidence'] = result_data['evidence']
                
                # Add defects if any
                if 'defects' in result_data:
                    test_run['defects'] = result_data['defects']
                
                # Add actual results for steps if provided
                if 'step_results' in result_data:
                    test_run['steps'] = result_data['step_results']
                
                results.append(test_run)
            
            # Update via Xray REST API
            url = f"{self.base_url}/rest/raven/1.0/import/execution"
            execution_data = {
                "testExecutionKey": execution_key,
                "tests": results
            }
            
            response = self.session.post(url, json=execution_data, timeout=self.request_timeout)
            self._handle_response_errors(response, f"update test execution {execution_key}")
            
            return json.dumps({
                'success': True,
                'execution_key': execution_key,
                'updated_tests': len(results),
                'message': f"Test execution {execution_key} updated successfully"
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to update test execution {execution_key}: {e}")
            raise
    
    def _handle_get_test_set(self, test_set_key: str, **kwargs) -> str:
        """Get test set details with all test cases."""
        
        if not test_set_key:
            raise XrayDataError("Test set key is required")
        
        # Check cache
        cache_key = f"ts_{test_set_key}"
        cached = self._get_from_cache(cache_key, 'test_set')
        if cached:
            return cached
        
        try:
            # Get test set issue
            url = f"{self.base_url}/rest/api/3/issue/{test_set_key}"
            response = self.session.get(url, timeout=self.request_timeout)
            self._handle_response_errors(response, test_set_key)
            
            test_set = response.json()
            
            # Get tests in the test set via JQL
            jql = f'"Test Sets" = {test_set_key}'
            search_url = f"{self.base_url}/rest/api/3/search"
            search_params = {
                "jql": jql,
                "fields": "key,summary,status,priority",
                "maxResults": 1000
            }
            
            response = self.session.get(search_url, params=search_params, timeout=self.request_timeout)
            self._handle_response_errors(response, "search tests in set")
            
            search_results = response.json()
            
            result = {
                'test_set': {
                    'key': test_set['key'],
                    'summary': test_set['fields']['summary'],
                    'description': self._extract_description(test_set['fields'].get('description'))
                },
                'tests': [
                    {
                        'key': issue['key'],
                        'summary': issue['fields']['summary'],
                        'status': issue['fields']['status']['name'],
                        'priority': issue['fields']['priority']['name']
                    }
                    for issue in search_results['issues']
                ],
                'total_tests': search_results['total']
            }
            
            # Cache the result
            self._add_to_cache(cache_key, json.dumps(result), 'test_set')
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to get test set {test_set_key}: {e}")
            raise
    
    def _handle_bulk_import(self, test_cases: List[Dict[str, Any]], project_key: str, **kwargs) -> str:
        """Bulk import multiple test cases."""
        
        if not test_cases:
            raise XrayDataError("No test cases provided for bulk import")
        
        if not project_key:
            raise XrayDataError("Project key is required for bulk import")
        
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_test = {
                executor.submit(
                    self._handle_create_test_case, project_key, test_data
                ): idx for idx, test_data in enumerate(test_cases)
            }
            
            for future in as_completed(future_to_test):
                idx = future_to_test[future]
                try:
                    result = future.result()
                    results.append(json.loads(result))
                except Exception as e:
                    logger.error(f"Failed to import test case {idx}: {e}")
                    errors.append({'index': idx, 'error': str(e)})
        
        return json.dumps({
            'imported': results,
            'errors': errors,
            'summary': {
                'total': len(test_cases),
                'successful': len(results),
                'failed': len(errors)
            }
        }, indent=2)
    
    def _handle_get_test_plan(self, test_plan_key: str, **kwargs) -> str:
        """Get test plan details with test execution status."""
        
        if not test_plan_key:
            raise XrayDataError("Test plan key is required")
        
        # Check cache
        cache_key = f"tp_{test_plan_key}"
        cached = self._get_from_cache(cache_key, 'test_plan')
        if cached:
            return cached
        
        try:
            # Get test plan issue
            url = f"{self.base_url}/rest/api/3/issue/{test_plan_key}"
            response = self.session.get(url, timeout=self.request_timeout)
            self._handle_response_errors(response, test_plan_key)
            
            test_plan = response.json()
            
            # Get test executions for this test plan
            jql = f'"Test Plan" = {test_plan_key}'
            search_url = f"{self.base_url}/rest/api/3/search"
            search_params = {
                "jql": jql,
                "fields": "key,summary,status",
                "maxResults": 100
            }
            
            response = self.session.get(search_url, params=search_params, timeout=self.request_timeout)
            self._handle_response_errors(response, "search test executions")
            
            executions = response.json()
            
            # Get test coverage statistics
            stats = self._get_test_plan_statistics(test_plan_key)
            
            result = {
                'test_plan': {
                    'key': test_plan['key'],
                    'summary': test_plan['fields']['summary'],
                    'description': self._extract_description(test_plan['fields'].get('description')),
                    'status': test_plan['fields']['status']['name']
                },
                'executions': [
                    {
                        'key': exec['key'],
                        'summary': exec['fields']['summary'],
                        'status': exec['fields']['status']['name']
                    }
                    for exec in executions['issues']
                ],
                'statistics': stats,
                'total_executions': executions['total']
            }
            
            # Cache the result
            self._add_to_cache(cache_key, json.dumps(result), 'test_plan')
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to get test plan {test_plan_key}: {e}")
            raise
    
    # Helper methods
    
    def _handle_response_errors(self, response: requests.Response, context: str):
        """Handle HTTP response errors."""
        if response.status_code == 401:
            raise XrayAuthenticationError("Invalid credentials or expired token")
        elif response.status_code == 403:
            raise XrayAuthenticationError(f"Insufficient permissions for {context}")
        elif response.status_code == 404:
            raise XrayDataError(f"Resource not found: {context}")
        elif response.status_code >= 500:
            raise XrayConnectionError(f"Xray server error: HTTP {response.status_code}")
        elif not response.ok:
            response.raise_for_status()
    
    def _format_test_case_data(self, data: Dict[str, Any], include_steps: bool,
                              include_attachments: bool, include_requirements: bool,
                              executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format test case data into structured response."""
        fields = data['fields']

        result = {
            'key': data['key'],            
            'summary': fields.get('summary', ''),
            'description': self._extract_description(fields.get('description')),
            'status': fields.get('status', {}).get('name', 'Unknown') if fields.get('status') else 'Unknown',
            'priority': fields.get('priority', {}).get('name', 'Unknown') if fields.get('priority') else 'Unknown',
            'test_type': fields.get(self.custom_fields['test_type'], 'Manual'),
            'labels': fields.get('labels', []),
            'created': fields.get('created'),
            'updated': fields.get('updated'),
            'reporter': fields.get('reporter', {}).get('displayName', 'Unknown') if fields.get('reporter') else 'Unknown',
            'assignee': fields.get('assignee', {}).get('displayName', 'Unassigned') if fields.get('assignee') else 'Unassigned'
        }
        
        # Add optional fields
        if fields.get('components'):
            result['components'] = [c['name'] for c in fields['components']]
        
        if fields.get('fixVersions'):
            result['fix_versions'] = [v['name'] for v in fields['fixVersions']]
        
        if include_steps and self.custom_fields['steps'] in fields:
            result['steps'] = self._extract_test_steps(fields[self.custom_fields['steps']])
        
        if fields.get(self.custom_fields['preconditions']):
            result['preconditions'] = fields[self.custom_fields['preconditions']]
        
        if fields.get(self.custom_fields['objective']):
            result['objective'] = fields[self.custom_fields['objective']]
        
        if include_attachments and fields.get('attachment'):
            result['attachments'] = self._extract_attachments(fields['attachment'])
        
        if include_requirements:
            result['requirements'] = self._extract_requirements(fields.get('issuelinks', []))
        
        if executions:
            result['executions'] = executions
        
        return result
    
    def _format_test_steps(self, steps: List[Union[TestStep, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format test steps for Xray API."""
        formatted_steps = []
        
        for step in steps:
            if isinstance(step, TestStep):
                step_data = step.dict()
            else:
                step_data = step
            
            formatted_step = {
                "action": step_data.get('action', ''),
                "data": step_data.get('data', ''),
                "result": step_data.get('expected_result', step_data.get('result', ''))
            }
            
            if 'attachments' in step_data:
                formatted_step['attachments'] = step_data['attachments']
            
            formatted_steps.append(formatted_step)
        
        return formatted_steps
    
    def _format_description(self, description: Optional[str]) -> Dict[str, Any]:
        """Format description for Jira ADF format."""
        if not description:
            return None
        
        # Simple ADF format
        return {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": description
                        }
                    ]
                }
            ]
        }
    
    def _extract_description(self, description_field: Any) -> str:
        """Extract description from various Jira formats."""
        if not description_field:
            return ""
        
        if isinstance(description_field, str):
            return description_field
        elif isinstance(description_field, dict):
            # Handle ADF format
            if 'content' in description_field:
                return self._extract_adf_text(description_field)
        
        return str(description_field)
    
    def _extract_adf_text(self, adf_content: Dict[str, Any]) -> str:
        """Extract plain text from Atlassian Document Format."""
        def extract_text_recursive(node):
            text_parts = []
            
            if isinstance(node, dict):
                if node.get('type') == 'text':
                    text_parts.append(node.get('text', ''))
                elif 'content' in node:
                    for child in node['content']:
                        text_parts.append(extract_text_recursive(child))
            elif isinstance(node, list):
                for item in node:
                    text_parts.append(extract_text_recursive(item))
            
            return ' '.join(filter(None, text_parts))
        
        return extract_text_recursive(adf_content).strip()
    
    def _extract_test_steps(self, steps_field: Any) -> List[Dict[str, str]]:
        """Extract and format test steps."""
        if not steps_field:
            return []
        
        if isinstance(steps_field, list):
            formatted_steps = []
            for i, step in enumerate(steps_field, 1):
                if isinstance(step, dict):
                    formatted_steps.append({
                        'index': i,
                        'action': step.get('action', ''),
                        'data': step.get('data', ''),
                        'expected_result': step.get('result', '')
                    })
            return formatted_steps
        
        return []
    
    def _extract_attachments(self, attachments_field: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract attachment information."""
        if not attachments_field:
            return []
        
        attachments = []
        for attachment in attachments_field[:20]:  # Limit to first 20
            if isinstance(attachment, dict):
                attachments.append({
                    'filename': attachment.get('filename', 'Unknown'),                    'size': attachment.get('size', 0),
                    'mimeType': attachment.get('mimeType', 'Unknown'),
                    'created': attachment.get('created', 'Unknown'),
                    'author': attachment.get('author', {}).get('displayName', 'Unknown') if attachment.get('author') else 'Unknown'
                })
        
        return attachments
    
    def _extract_requirements(self, issue_links: List[Dict[str, Any]]) -> List[Dict[str, str]]:        
        """Extract linked requirements from issue links."""
        requirements = []
        
        for link in issue_links:
            link_type = link.get('type')
            if link_type and link_type.get('name') in ['Tests', 'Requirement']:
                if 'outwardIssue' in link:
                    issue = link['outwardIssue']
                    requirements.append({
                        'key': issue['key'],
                        'summary': issue['fields']['summary'],
                        'status': issue['fields']['status']['name']
                    })
                elif 'inwardIssue' in link:
                    issue = link['inwardIssue']
                    requirements.append({
                        'key': issue['key'],
                        'summary': issue['fields']['summary'],
                        'status': issue['fields']['status']['name']
                    })
        
        return requirements
    
    def _get_test_executions(self, test_case_key: str) -> List[Dict[str, Any]]:
        """Get test executions for a test case."""
        try:
            # Use JQL to find test executions containing this test
            jql = f'issue in testExecutionTests("{test_case_key}")'
            url = f"{self.base_url}/rest/api/3/search"
            params = {
                "jql": jql,
                "fields": "key,summary,status,created",
                "maxResults": 10
            }
            
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            if response.ok:
                data = response.json()
                return [
                    {
                        'key': issue['key'],
                        'summary': issue['fields']['summary'],
                        'status': issue['fields']['status']['name'],
                        'created': issue['fields']['created']
                    }
                    for issue in data['issues']
                ]
        except Exception as e:
            logger.warning(f"Failed to get test executions for {test_case_key}: {e}")
        
        return []
    
    def _get_issue_basic_info(self, issue_key: str) -> Dict[str, Any]:
        """Get basic issue information."""
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        params = {"fields": "project,issuetype,summary"}
        
        response = self.session.get(url, params=params, timeout=self.request_timeout)
        self._handle_response_errors(response, issue_key)
        
        return response.json()
    
    def _link_requirements(self, test_key: str, requirement_keys: List[str]):
        """Link requirements to a test case."""
        for req_key in requirement_keys:
            try:
                link_data = {
                    "type": {"name": "Tests"},
                    "inwardIssue": {"key": test_key},
                    "outwardIssue": {"key": req_key}
                }
                
                url = f"{self.base_url}/rest/api/3/issueLink"
                response = self.session.post(url, json=link_data, timeout=self.request_timeout)
                response.raise_for_status()
                
            except Exception as e:
                logger.warning(f"Failed to link requirement {req_key} to {test_key}: {e}")
    
    def _add_tests_to_execution(self, execution_key: str, test_keys: List[str]):
        """Add test cases to a test execution."""
        try:
            # Xray Cloud API endpoint
            url = f"{self.base_url}/rest/raven/1.0/api/testexec/{execution_key}/test"
            
            response = self.session.post(url, json={"add": test_keys}, timeout=self.request_timeout)
            response.raise_for_status()
            
        except Exception as e:
            logger.warning(f"Failed to add tests to execution {execution_key}: {e}")
    
    def _get_test_plan_statistics(self, test_plan_key: str) -> Dict[str, Any]:
        """Get test plan execution statistics."""
        try:
            # This would typically call Xray's specific statistics endpoint
            # Placeholder implementation
            return {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'executing': 0,
                'todo': 0,
                'blocked': 0
            }
        except Exception as e:
            logger.warning(f"Failed to get test plan statistics: {e}")
            return {}
    
    # Cache management methods
    
    def _get_from_cache(self, key: str, cache_type: str) -> Optional[str]:
        """Get item from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            ttl = self.cache_ttl.get(cache_type, 300)
            if time.time() - entry['timestamp'] < ttl:
                logger.debug(f"Cache hit for {key}")
                return entry['data']
            else:
                del self.cache[key]
        return None
    
    def _add_to_cache(self, key: str, data: str, cache_type: str):
        """Add item to cache."""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'type': cache_type
        }
    
    def clear_cache(self):
        """Clear the entire cache."""
        self.cache.clear()
        logger.info("Xray API cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'total_entries': len(self.cache),
            'entries_by_type': {},
            'cache_size_bytes': 0
        }
        
        for key, entry in self.cache.items():
            cache_type = entry.get('type', 'unknown')
            stats['entries_by_type'][cache_type] = stats['entries_by_type'].get(cache_type, 0) + 1
            stats['cache_size_bytes'] += len(entry['data'])
        
        return stats
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()


# Backward compatibility
XrayAPITool = EnhancedXrayAPITool