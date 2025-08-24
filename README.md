# Backend API Test Generator with Enhanced Bitbucket Integration

A powerful CrewAI-based tool that automatically generates comprehensive API tests from Jira requirements with advanced Bitbucket repository analysis. This enhanced system analyzes your backend code (FastAPI/Django), compares existing tests with requirements, identifies gaps, and creates or modifies pytest-based test suites with full coverage.

## Enhanced Features

- **Bitbucket Integration**: Clone and analyze Bitbucket repositories for existing test coverage
- **Jira Integration**: Extract API testing requirements directly from Jira tickets
- **Test Gap Analysis**: Compare existing tests with requirements to identify coverage gaps
- **Multi-Framework Support**: Works with FastAPI and Django applications
- **Comprehensive Test Generation**: Creates unit tests, integration tests, and API validation tests
- **AI-Powered Analysis**: Uses AWS Bedrock Claude models for intelligent code analysis
- **Pull Request Creation**: Automatically creates PRs with generated/modified tests
- **Best Practices**: Generates tests following pytest best practices with proper fixtures
- **Quality Validation**: Validates generated tests for syntax, imports, and quality

## Enhanced Architecture

The enhanced system uses 7 specialized AI agents working through 11 comprehensive tasks:

### Agents:
1. **Bitbucket Analyst**: Clones and analyzes Bitbucket repositories
2. **Test Gap Analyzer**: Identifies gaps between existing tests and requirements
3. **Jira Requirements Extractor**: Extracts API testing requirements from Jira tickets
4. **Test Comparison Expert**: Compares existing tests with extracted requirements
5. **API Test Generator**: Generates high-quality pytest code for API testing
6. **Test Validator**: Validates and ensures quality of generated tests
7. **PR Creator**: Creates pull requests with generated/modified tests

### Workflow Tasks:
1. Clone Bitbucket repository
2. Analyze repository structure and existing tests
3. Extract requirements from Jira ticket
4. Analyze existing test coverage
5. Compare tests with requirements
6. Identify test gaps and coverage issues
7. Generate or modify tests based on analysis
8. Organize test files and structure
9. Validate generated tests
10. Create pull request with changes
11. Generate comprehensive final report

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system.

### Option 1: Using pip

```bash
pip install -r requirements.txt
```

### Option 2: Using UV (Recommended)

```bash
pip install uv
uv pip install -r requirements.txt
```

### Option 3: Using CrewAI CLI

```bash
crewai install
```

## Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Update the `.env` file with your configuration:
   ```bash
   # AWS Configuration (for AI models)
   AWS_REGION=us-west-2
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key

   # Jira Configuration
   JIRA_URL=https://your-company.atlassian.net
   JIRA_TOKEN=your_jira_api_token

   # Bitbucket Configuration (Enhanced Workflow)
   BITBUCKET_USERNAME=your_bitbucket_username
   BITBUCKET_APP_PASSWORD=your_bitbucket_app_password

   # API Testing Configuration
   API_BASE_URL=http://localhost:8000
   TEST_AUTH_TOKEN=test-auth-token-here
   
   # Test Environment
   TEST_ENV=local
   COVERAGE_THRESHOLD=80
   ```

3. Validate your configuration:
   ```bash
   python validate_workflow.py
   ```
   cp .env.example .env
   ```

2. Configure your environment variables in `.env`:
   ```bash
   # AWS Configuration
   AWS_REGION=us-west-2
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key

   # Jira Configuration
   JIRA_URL=https://your-company.atlassian.net
   JIRA_TOKEN=your_jira_api_token

   # API Testing Configuration
   API_BASE_URL=http://localhost:8000
   TEST_AUTH_TOKEN=test-auth-token-here
   TEST_ADMIN_TOKEN=test-admin-token-here
   ```

## Enhanced Workflow Usage

### Basic Enhanced Workflow

Generate or modify tests from Jira requirements with Bitbucket integration:

```bash
python -m test_generation_with_claude.main \
  --jira-key PROJ-123 \
  --bitbucket-repo https://bitbucket.org/workspace/repo \
  --backend-path ./src \
  --framework fastapi
```

### Advanced Enhanced Workflow

```bash
python -m test_generation_with_claude.main \
  --jira-key PROJ-123 \
  --bitbucket-repo https://bitbucket.org/workspace/repo \
  --bitbucket-branch develop \
  --backend-path ./my-api/src \
  --framework django \
  --action auto \
  --output-dir ./tests \
  --workspace-dir ./workspace \
  --coverage-threshold 90 \
  --test-pattern "test_*.py" \
  --verbose
```

### Workflow Control Options

```bash
# Dry run (analysis only, no test generation)
python -m test_generation_with_claude.main \
  --jira-key PROJ-123 \
  --bitbucket-repo https://bitbucket.org/workspace/repo \
  --dry-run

# Skip repository cloning (use existing workspace)
python -m test_generation_with_claude.main \
  --jira-key PROJ-123 \
  --bitbucket-repo https://bitbucket.org/workspace/repo \
  --skip-clone

# Skip pull request creation
python -m test_generation_with_claude.main \
  --jira-key PROJ-123 \
  --bitbucket-repo https://bitbucket.org/workspace/repo \
  --skip-pr
```

### Enhanced Parameters

#### Required Parameters
- `--jira-key`: Jira ticket key (e.g., PROJ-123)
- `--bitbucket-repo`: Bitbucket repository URL (e.g., https://bitbucket.org/workspace/repo)

#### Core Configuration
- `--bitbucket-branch`: Git branch to work on (default: main)
- `--backend-path`: Path to backend code to analyze (default: ./src)
- `--framework`: Backend framework - 'fastapi' or 'django' (default: fastapi)
- `--action`: Test action - 'modify', 'add', or 'auto' (default: auto)

#### Directory Options
- `--output-dir`: Output directory for tests (default: ./tests)
- `--workspace-dir`: Workspace directory for temporary files (default: ./workspace)

#### Analysis Options
- `--coverage-threshold`: Minimum coverage threshold percentage (default: 80)
- `--test-pattern`: Pattern for test files to analyze (default: test_*.py)

#### Workflow Control
- `--skip-clone`: Skip cloning repository (use existing workspace)
- `--skip-pr`: Skip creating pull request
- `--dry-run`: Perform analysis only, don't generate or modify tests
- `--verbose`: Enable verbose output

### Help and Information

```bash
# Get detailed help
python -m test_generation_with_claude.main --help

# Validate workflow setup
python validate_workflow.py
```

## Generated Output

The tool generates:

### Test Files
- `tests/api/test_*.py`: Individual endpoint test files
- `tests/integration/test_workflows.py`: Integration test workflows
- `tests/conftest.py`: Pytest fixtures and configuration

### Reports
- `output/requirements_{jira_key}.json`: Extracted requirements
- `output/api_analysis_{jira_key}.json`: API endpoint analysis
- `output/test_scenarios_{jira_key}.json`: Test scenario design
- `output/validation_report.json`: Test validation results
- `output/test_report.html`: Test execution report

## Example Generated Test

```python
"""
API tests for /api/users
Generated automatically from requirements
"""

import pytest
import httpx
from typing import Dict, Any

class TestUsersEndpoint:
    """Test class for /api/users endpoint"""
    
    @pytest.fixture
    def endpoint_url(self, base_url):
        return f"{base_url}/api/users"
    
    def test_create_user_success(self, api_client, endpoint_url, auth_headers):
        """
        Test successful user creation
        Test Type: Happy Path
        """
        # Arrange
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "first_name": "Test",
            "last_name": "User"
        }
        
        # Act
        response = api_client.post(
            endpoint_url,
            json=user_data,
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 201
        assert response.headers["content-type"] == "application/json"
        
        response_data = response.json()
        assert response_data["username"] == user_data["username"]
        assert "id" in response_data
```

## Project Structure

```
backend-api-test-generator/
├── config/
│   ├── agents.yaml        # Agent definitions
│   └── tasks.yaml         # Task definitions
├── src/
│   ├── test_generation_with_claude/
│   │   ├── __init__.py
│   │   ├── crew.py        # Main crew implementation
│   │   ├── tools/         # Custom tools
│   │   │   ├── api_analyzer.py
│   │   │   ├── jira_tools.py
│   │   │   └── test_generator.py
│   │   └── templates/     # Test templates
│   │       ├── api_test_base.py
│   │       └── fixtures.py
├── tests/                 # Generated tests output
├── output/                # Analysis reports and results
├── main.py               # Entry point
├── requirements.txt
└── .env.example
```

## Customization

### Adding New Tools

Create custom tools in `src/test_generation_with_claude/tools/`:

```python
from crewai_tools import BaseTool

class CustomAnalyzerTool(BaseTool):
    name: str = "Custom Analyzer"
    description: str = "Custom tool description"
    
    def _run(self, input_data: str) -> str:
        # Your custom logic here
        return "result"
```

### Modifying Agents

Edit `config/agents.yaml` to customize agent behavior:

```yaml
my_custom_agent:
  role: "Custom API Analyst"
  goal: "Analyze custom API patterns"
  backstory: "You are an expert in custom API analysis..."
  tools:
    - CustomAnalyzerTool
  max_iter: 3
  verbose: true
```

### Adding New Tasks

Edit `config/tasks.yaml` to add custom tasks:

```yaml
custom_analysis_task:
  description: "Perform custom analysis on the API"
  expected_output: "Custom analysis report"
  agent: my_custom_agent
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code  
flake8 src/ tests/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure AWS credentials are properly configured
2. **Jira Access**: Verify Jira URL and token have correct permissions
3. **Python Version**: Ensure Python 3.10+ is being used
4. **Dependencies**: Run `pip install -r requirements.txt` if imports fail

### Debug Mode

Run with verbose output:

```bash
python main.py --jira-key PROJ-123 --backend-path ./src --framework fastapi --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, questions, or feedback:
- Create an issue in the GitHub repository
- Check the [CrewAI documentation](https://docs.crewai.com)
- Review the troubleshooting section above
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
