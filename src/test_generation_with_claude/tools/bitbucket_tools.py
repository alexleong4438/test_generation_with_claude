"""
Enhanced Bitbucket integration tools with built-in error handling
"""

import os
import subprocess
import requests
import time
import json
from pathlib import Path
import tempfile
import shutil
from crewai.tools import BaseTool
from typing import Dict, Any, Optional, List
import git
import logging
from functools import wraps
from time import sleep
import random

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom exception classes
class ToolError(Exception):
    """Base exception for tool errors"""
    pass

class ToolConnectionError(ToolError):
    """Connection-related errors"""
    pass

class ToolAuthenticationError(ToolError):
    """Authentication-related errors"""
    pass

class ToolDataError(ToolError):
    """Data validation errors"""
    pass

class ToolConfigurationError(ToolError):
    """Configuration errors"""
    pass


# Retry decorator
def retry_on_exception(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"Max retries ({max_attempts}) reached for {func.__name__}")
                        raise
                    
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}")
                    logger.info(f"Retrying in {current_delay} seconds...")
                    
                    # Add jitter to prevent thundering herd
                    jittered_delay = current_delay * (1 + random.uniform(-0.1, 0.1))
                    sleep(jittered_delay)
                    
                    current_delay *= backoff
                    attempt += 1
            
            return None
        return wrapper
    return decorator


# Performance timer context manager
class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        logger.info(f"{self.operation_name} completed in {elapsed:.2f} seconds")
        return False


class EnhancedBitbucketCloneTool(BaseTool):
    """Enhanced tool for cloning Bitbucket repositories with error handling and retry logic"""
    
    name: str = "Enhanced Bitbucket Clone Tool"
    description: str = "Clone a Bitbucket repository to a local workspace directory with enhanced reliability"
    workspace_dir: str = "./workspace"
    api_token: Optional[str] = None
    username: Optional[str] = None
    clone_cache: Dict[str, Any] = {}  # Cache for cloned repositories
    
    def __init__(self, workspace_dir: str = "./workspace", api_token: str = None, 
                 username: str = None, **kwargs):
        super().__init__(**kwargs)
        self.workspace_dir = workspace_dir
        self.clone_cache = {}
        # API token is preferred over username/password
        self.api_token = api_token or os.getenv("BITBUCKET_API_TOKEN")
        # Username is only needed for older auth methods
        self.username = username or os.getenv("BITBUCKET_USERNAME")
    
    def _run(self, repo_url: str, branch: str = "main", target_dir: Optional[str] = None, 
             force_fresh: bool = False, use_auth: bool = True) -> str:
        """
        Enhanced clone method with comprehensive error handling
        
        Args:
            repo_url: Bitbucket repository URL
            branch: Git branch to clone (default: main)
            target_dir: Target directory for cloning (optional)
            force_fresh: Force fresh clone even if cached (default: False)
            use_auth: Use authentication if credentials are available (default: True)
        
        Returns:
            JSON string with clone status and repository information
        """
        
        if not repo_url:
            raise ToolDataError("Repository URL cannot be empty")
        
        with PerformanceTimer(f"bitbucket_clone_{repo_url.split('/')[-1]}"):
            try:
                # Check cache first
                cache_key = f"{repo_url}_{branch}"
                if not force_fresh and cache_key in self.clone_cache:
                    cache_entry = self.clone_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < 3600:  # 1 hour cache
                        logger.debug(f"Using cached clone for {repo_url}")
                        return json.dumps(cache_entry['result'], indent=2)
                
                # Perform clone with retry
                result = self._clone_repository_with_retry(repo_url, branch, target_dir, use_auth)
                
                # Cache the result
                self.clone_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                logger.error(f"Failed to clone repository {repo_url}: {e}")
                error_result = {
                    "success": False,
                    "error": str(e),
                    "repo_url": repo_url,
                    "branch": branch,
                    "timestamp": time.time()
                }
                return json.dumps(error_result, indent=2)

    @retry_on_exception(max_attempts=3, delay=2.0, exceptions=(git.exc.GitCommandError, ToolConnectionError))
    def _clone_repository_with_retry(self, repo_url: str, branch: str, target_dir: Optional[str], use_auth: bool = True) -> Dict[str, Any]:
        """Clone repository with retry logic"""
        try:
            if target_dir is None:
                # Extract repo name from URL
                repo_name = repo_url.split('/')[-1].replace('.git', '')
                target_dir = Path(self.workspace_dir) / "bitbucket_repo" / repo_name
            else:
                target_dir = Path(target_dir)
            
            # Validate target directory
            if not target_dir.parent.exists():
                target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Clean existing directory if needed
            if target_dir.exists() and any(target_dir.iterdir()):
                logger.info(f"Cleaning existing directory: {target_dir}")
                shutil.rmtree(target_dir)
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare clone URL with authentication if needed
            clone_url = repo_url
            if use_auth and self.username and self.api_token:
                # Use API token for authentication (preferred method)
                if repo_url.startswith('https://') and '@' not in repo_url:
                    # Extract the domain and path
                    url_parts = repo_url[8:].split('/', 1)
                    domain = url_parts[0]
                    path = url_parts[1] if len(url_parts) > 1 else ''
                    
                    # Use token as username (this format works based on diagnostic)
                    clone_url = f"https://{self.username}:{self.api_token}@{domain}/{path}"
                    logger.info(f"Using API token authentication for cloning")
            
            logger.info(f"Cloning {repo_url} (branch: {branch}) to {target_dir}")
            
            # Clone repository with timeout and error handling
            try:
                # Set up environment for git command with timeout
                env = os.environ.copy()
                env['GIT_HTTP_LOW_SPEED_LIMIT'] = '1000'  # 1KB/s minimum speed
                env['GIT_HTTP_LOW_SPEED_TIME'] = '300'     # Allow 5 minutes of slow speed
                
                repo = git.Repo.clone_from(
                    clone_url,
                    target_dir,
                    branch=branch,
                    depth=1,  # Shallow clone for faster operation
                    env=env   # Pass environment variables
                )
            except git.exc.GitCommandError as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    raise ToolConnectionError(f"Clone operation timed out: {e}")
                elif "authentication" in error_msg.lower() or "permission" in error_msg.lower():
                    auth_hint = ""
                    if not self.api_token:
                        auth_hint = " (No API token provided. Set BITBUCKET_API_TOKEN environment variable)"
                    raise ToolAuthenticationError(f"Authentication failed{auth_hint}: {e}")
                elif "not found" in error_msg.lower():
                    raise ToolDataError(f"Repository not found: {e}")
                else:
                    raise ToolConnectionError(f"Git clone failed: {e}")
            
            # Get repository information
            repo_info = self._extract_repository_info(repo, target_dir, repo_url, branch)
            
            logger.info(f"Successfully cloned {repo_url} to {target_dir}")
            return repo_info
            
        except Exception as e:
            logger.error(f"Clone operation failed: {e}")
            # Clean up on failure
            if 'target_dir' in locals() and target_dir.exists():
                try:
                    shutil.rmtree(target_dir)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup directory {target_dir}: {cleanup_error}")
            raise

    def _extract_repository_info(self, repo: git.Repo, target_dir: Path, repo_url: str, branch: str) -> Dict[str, Any]:
        """Extract comprehensive repository information"""
        try:
            # Count different file types
            file_counts = self._count_files_by_type(target_dir)
            
            # Find test files
            test_files = self._find_test_files(target_dir)
            
            # Basic repo info
            repo_info = {
                "success": True,
                "repo_path": str(target_dir),
                "repo_url": repo_url,
                "branch": branch,
                "commit_hash": repo.head.commit.hexsha,
                "short_commit_hash": repo.head.commit.hexsha[:8],
                "commit_message": repo.head.commit.message.strip(),
                "author": str(repo.head.commit.author),
                "commit_date": repo.head.commit.committed_datetime.isoformat(),
                "files_count": file_counts['total'],
                "file_types": file_counts['by_type'],
                "test_files": test_files,
                "has_tests": len(test_files) > 0,
                "clone_timestamp": time.time(),
                "message": f"Successfully cloned {repo_url} (branch: {branch}) to {target_dir}"
            }
            
            # Try to detect project type
            project_info = self._detect_project_type(target_dir)
            repo_info.update(project_info)
            
            return repo_info
            
        except Exception as e:
            logger.warning(f"Failed to extract complete repository info: {e}")
            # Return minimal info if extraction fails
            return {
                "success": True,
                "repo_path": str(target_dir),
                "repo_url": repo_url,
                "branch": branch,
                "commit_hash": repo.head.commit.hexsha[:8],
                "message": f"Cloned {repo_url} with limited info extraction",
                "warning": f"Info extraction failed: {e}"
            }

    def _count_files_by_type(self, directory: Path) -> Dict[str, Any]:
        """Count files by type in the repository"""
        counts = {'total': 0, 'by_type': {}}
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    counts['total'] += 1
                    extension = file_path.suffix.lower()
                    if extension:
                        counts['by_type'][extension] = counts['by_type'].get(extension, 0) + 1
                    else:
                        counts['by_type']['no_extension'] = counts['by_type'].get('no_extension', 0) + 1
        except Exception as e:
            logger.warning(f"Failed to count files: {e}")
            
        return counts

    def _find_test_files(self, directory: Path) -> List[str]:
        """Find test files in the repository"""
        test_files = []
        test_patterns = ['test_*.py', '*_test.py', 'test*.py', '*test*.py']
        
        try:
            for pattern in test_patterns:
                test_files.extend([str(f.relative_to(directory)) for f in directory.rglob(pattern)])
            
            # Also look in common test directories
            test_dirs = ['tests', 'test', 'testing', '__tests__']
            for test_dir in test_dirs:
                test_path = directory / test_dir
                if test_path.exists():
                    test_files.extend([str(f.relative_to(directory)) for f in test_path.rglob("*.py")])
            
            # Remove duplicates and sort
            test_files = sorted(list(set(test_files)))
            
        except Exception as e:
            logger.warning(f"Failed to find test files: {e}")
            
        return test_files

    def _detect_project_type(self, directory: Path) -> Dict[str, Any]:
        """Detect project type and framework"""
        project_info = {
            "project_type": "unknown",
            "framework": "unknown",
            "has_requirements": False,
            "has_docker": False,
            "has_ci": False
        }
        
        try:
            # Check for Python project files
            if (directory / "requirements.txt").exists() or (directory / "pyproject.toml").exists():
                project_info["project_type"] = "python"
                project_info["has_requirements"] = True
                
                # Detect framework
                if (directory / "manage.py").exists():
                    project_info["framework"] = "django"
                elif any(directory.rglob("*fastapi*")):
                    project_info["framework"] = "fastapi"
                elif any(directory.rglob("*flask*")):
                    project_info["framework"] = "flask"
            
            # Check for Docker
            if (directory / "Dockerfile").exists() or (directory / "docker-compose.yml").exists():
                project_info["has_docker"] = True
            
            # Check for CI/CD
            ci_patterns = [".github/workflows", ".gitlab-ci.yml", "bitbucket-pipelines.yml", ".travis.yml"]
            for pattern in ci_patterns:
                if (directory / pattern).exists():
                    project_info["has_ci"] = True
                    break
                    
        except Exception as e:
            logger.warning(f"Failed to detect project type: {e}")
            
        return project_info

    def clear_cache(self):
        """Clear the clone cache"""
        self.clone_cache.clear()
        logger.debug("Bitbucket clone cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.clone_cache),
            'cached_repos': list(self.clone_cache.keys())
        }

    def set_credentials(self, api_token: str = None, username: str = None):
        """
        Set Bitbucket credentials for private repository access
        
        Args:
            api_token: Bitbucket API token (Personal Access Token)
            username: Bitbucket username (optional, only for legacy auth)
        """
        if api_token:
            self.api_token = api_token
            logger.info("Bitbucket API token updated")
        if username:
            self.username = username
            logger.info("Bitbucket username updated")
    
    def has_credentials(self) -> bool:
        """Check if credentials are available"""
        return bool(self.api_token)
    
    def test_authentication(self, repo_url: str) -> bool:
        """
        Test if authentication works for a repository without prompts
        
        Args:
            repo_url: Repository URL to test
            
        Returns:
            True if authentication works, False otherwise
        """
        if not self.api_token or not self.username:
            logger.warning("No credentials available for authentication test")
            return False
            
        try:
            # Prepare authenticated URL
            if repo_url.startswith('https://') and '@' not in repo_url:
                url_parts = repo_url[8:].split('/', 1)
                domain = url_parts[0]
                path = url_parts[1] if len(url_parts) > 1 else ''
                test_url = f"https://{self.username}:{self.api_token}@{domain}/{path}"
            else:
                test_url = repo_url
            
            # Test with git ls-remote (faster than clone)
            env = os.environ.copy()
            env['GIT_TERMINAL_PROMPT'] = '0'
            env['GIT_ASKPASS'] = 'echo'
            
            result = subprocess.run(
                ['git', 'ls-remote', '--heads', test_url],
                capture_output=True,
                text=True,
                env=env,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("Authentication test successful")
                return True
            else:
                logger.error(f"Authentication test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication test error: {e}")
            return False
    
    def test_authentication(self, repo_url: str) -> bool:
        """
        Test if authentication works for a repository without prompts
        
        Args:
            repo_url: Repository URL to test
            
        Returns:
            True if authentication works, False otherwise
        """
        if not self.api_token:
            logger.warning("No API token available for authentication test")
            return False
            
        try:
            # Prepare authenticated URL
            if repo_url.startswith('https://') and '@' not in repo_url:
                url_parts = repo_url[8:].split('/', 1)
                domain = url_parts[0]
                path = url_parts[1] if len(url_parts) > 1 else ''
                test_url = f"https://{self.api_token}@{domain}/{path}"
            else:
                test_url = repo_url
            
            # Test with git ls-remote (faster than clone)
            env = os.environ.copy()
            env['GIT_TERMINAL_PROMPT'] = '0'
            env['GIT_ASKPASS'] = 'echo'
            
            result = subprocess.run(
                ['git', 'ls-remote', '--heads', test_url],
                capture_output=True,
                text=True,
                env=env,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("Authentication test successful")
                return True
            else:
                logger.error(f"Authentication test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication test error: {e}")
            return False


class EnhancedBitbucketPRTool(BaseTool):
    """Enhanced tool for creating pull requests on Bitbucket with error handling"""
    
    name: str = "Enhanced Bitbucket PR Tool"
    description: str = "Create a pull request on Bitbucket with generated/modified tests with enhanced reliability"
    api_token: Optional[str] = None
    base_url: str = "https://api.bitbucket.org/2.0"
    
    def __init__(self, api_token: str = None, **kwargs):
        super().__init__(**kwargs)
        self.api_token = api_token or os.getenv("BITBUCKET_API_TOKEN")
        self.base_url = "https://api.bitbucket.org/2.0"
    
    def _run(self, repo_url: str, title: str, description: str, 
             source_branch: str, target_branch: str = "main",
             repo_path: str = None) -> Dict[str, Any]:
        """
        Create a pull request on Bitbucket
        
        Args:
            repo_url: Bitbucket repository URL
            title: Pull request title
            description: Pull request description
            source_branch: Source branch name
            target_branch: Target branch name (default: main)
            repo_path: Local repository path for committing changes
        
        Returns:
            Dictionary with PR creation status and URL
        """
        try:
            if not self.api_token:
                return {
                    "success": False,
                    "error": "Bitbucket API token not provided",
                    "message": "Please set BITBUCKET_API_TOKEN environment variable"
                }
            
            # Extract workspace and repo name from URL
            # Expected format: https://bitbucket.org/workspace/repo
            url_parts = repo_url.replace('.git', '').split('/')
            workspace = url_parts[-2]
            repo_name = url_parts[-1]
            
            # If repo_path is provided, commit and push changes
            if repo_path and Path(repo_path).exists():
                try:
                    repo = git.Repo(repo_path)
                    
                    # Create and checkout new branch
                    try:
                        repo.git.checkout('-b', source_branch)
                    except git.exc.GitCommandError:
                        # Branch might already exist
                        repo.git.checkout(source_branch)
                    
                    # Add all changes
                    repo.git.add('--all')
                    
                    # Check if there are changes to commit
                    if repo.is_dirty() or repo.untracked_files:
                        repo.git.commit('-m', f"Add/modify tests for {title}")
                        
                        # Set up remote with API token
                        if repo_url.startswith('https://') and '@' not in repo_url:
                            url_parts = repo_url[8:].split('/', 1)
                            domain = url_parts[0]
                            path = url_parts[1] if len(url_parts) > 1 else ''
                            # Use token directly as username (proven to work)
                            remote_url = f"https://{self.api_token}@{domain}/{path}"
                        else:
                            remote_url = repo_url
                            
                        try:
                            origin = repo.remote('origin')
                            origin.set_url(remote_url)
                        except git.exc.GitCommandError:
                            origin = repo.create_remote('origin', remote_url)
                        
                        # Push branch
                        origin.push(source_branch)
                    
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Git operations failed: {str(e)}",
                        "message": f"Failed to commit and push changes: {str(e)}"
                    }
            
            # Create pull request via API
            pr_data = {
                "title": title,
                "description": description,
                "source": {
                    "branch": {
                        "name": source_branch
                    }
                },
                "destination": {
                    "branch": {
                        "name": target_branch
                    }
                }
            }
            
            # Create pull request via API with Bearer token authentication
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            pr_url = f"{self.base_url}/repositories/{workspace}/{repo_name}/pullrequests"
            
            response = requests.post(pr_url, json=pr_data, headers=headers)
            
            if response.status_code == 201:
                pr_info = response.json()
                return {
                    "success": True,
                    "pr_id": pr_info["id"],
                    "pr_url": pr_info["links"]["html"]["href"],
                    "title": title,
                    "source_branch": source_branch,
                    "target_branch": target_branch,
                    "message": f"Pull request created successfully: {pr_info['links']['html']['href']}"
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code} - {response.text}",
                    "message": f"Failed to create pull request: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "message": f"Failed to create pull request: {str(e)}"
            }


class BitbucketAnalyzerTool(BaseTool):
    """Tool for analyzing Bitbucket repository structure and content"""
    
    name: str = "BitbucketAnalyzerTool" 
    description: str = "Analyze Bitbucket repository structure, find test files, and extract metadata"
    
    def _run(self, repo_path: str, test_pattern: str = "test_*.py") -> Dict[str, Any]:
        """
        Analyze repository structure and content
        
        Args:
            repo_path: Path to the cloned repository
            test_pattern: Pattern for identifying test files
        
        Returns:
            Dictionary with repository analysis results
        """
        try:
            repo_path = Path(repo_path)
            
            if not repo_path.exists():
                return {
                    "success": False,
                    "error": f"Repository path does not exist: {repo_path}",
                    "message": "Repository path not found"
                }
            
            # Find all Python files
            python_files = list(repo_path.rglob("*.py"))
            
            # Find test files based on pattern
            import fnmatch
            test_files = []
            for py_file in python_files:
                if fnmatch.fnmatch(py_file.name, test_pattern) or "test" in py_file.parts:
                    test_files.append(py_file)
            
            # Find configuration files
            config_files = []
            config_patterns = ["*.yaml", "*.yml", "*.json", "*.toml", "*.ini", "*.cfg"]
            for pattern in config_patterns:
                config_files.extend(repo_path.rglob(pattern))
            
            # Find documentation files
            doc_files = []
            doc_patterns = ["README*", "*.md", "*.rst", "*.txt"]
            for pattern in doc_patterns:
                doc_files.extend(repo_path.rglob(pattern))
            
            # Directory structure analysis
            directories = [d for d in repo_path.rglob("*") if d.is_dir()]
            
            analysis_result = {
                "success": True,
                "repo_path": str(repo_path),
                "total_files": len(list(repo_path.rglob("*"))),
                "python_files": {
                    "count": len(python_files),
                    "files": [str(f.relative_to(repo_path)) for f in python_files]
                },
                "test_files": {
                    "count": len(test_files),
                    "files": [str(f.relative_to(repo_path)) for f in test_files]
                },
                "config_files": {
                    "count": len(config_files),
                    "files": [str(f.relative_to(repo_path)) for f in config_files]
                },
                "documentation": {
                    "count": len(doc_files),
                    "files": [str(f.relative_to(repo_path)) for f in doc_files]
                },
                "directories": {
                    "count": len(directories),
                    "structure": [str(d.relative_to(repo_path)) for d in directories]
                },
                "has_tests": len(test_files) > 0,
                "has_docs": len(doc_files) > 0,
                "message": f"Repository analysis completed. Found {len(test_files)} test files."
            }
            
            return analysis_result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis error: {str(e)}",
                "message": f"Failed to analyze repository: {str(e)}"
            }


# Example usage for private repositories
if __name__ == "__main__":
    """
    Example of how to use the Enhanced Bitbucket tools with private repositories
    """
    
    # Method 1: Using environment variables (recommended)
    # Set this in your environment:
    # export BITBUCKET_API_TOKEN="your_personal_access_token"
    
    # Create clone tool - will automatically use env var if set
    clone_tool = EnhancedBitbucketCloneTool(workspace_dir="./my_workspace")
    
    # Method 2: Pass API token directly
    # clone_tool = EnhancedBitbucketCloneTool(
    #     workspace_dir="./my_workspace",
    #     api_token="your_personal_access_token"
    # )
    
    # Method 3: Set credentials after initialization
    # clone_tool.set_credentials(api_token="your_personal_access_token")
    
    # Clone a private repository
    result = clone_tool._run(
        repo_url="https://bitbucket.org/yourworkspace/private-repo.git",
        branch="main"
    )
    
    print("Clone result:")
    print(result)
    
    # Create PR tool with API token
    pr_tool = EnhancedBitbucketPRTool()  # Will use env var
    
    # Analyze repository
    analyzer = BitbucketAnalyzerTool()
    
    # Example workflow:
    # 1. Clone repository
    # 2. Analyze structure
    # 3. Make changes
    # 4. Create pull request
    
    # How to create a Bitbucket API Token:
    # 1. Log into Bitbucket
    # 2. Click your avatar → Personal settings
    # 3. Click "Access tokens" in the left sidebar
    # 4. Click "Create a token"
    # 5. Give it a name and select scopes:
    #    - Repositories: Read (for cloning)
    #    - Repositories: Write (for pushing)
    #    - Pull requests: Write (for creating PRs)
    # 6. Click "Create" and save the token securely