"""
Bitbucket Authentication Diagnostic Tool
Helps diagnose and fix authentication issues
"""

import os
import subprocess
import requests
from getpass import getpass
import git
from pathlib import Path
import tempfile
import shutil

from bitbucket_tools import EnhancedBitbucketCloneTool


class BitbucketAuthDiagnostic:
    """Diagnostic tool for Bitbucket authentication issues"""
    
    def __init__(self):
        self.api_token = None
        self.test_repo = "https://bitbucket.org/f-secure/sbp_api.git"
        
    def run_diagnostics(self):
        """Run all diagnostic tests"""
        print("=" * 60)
        print("Bitbucket Authentication Diagnostic Tool")
        print("=" * 60)
        print()
        
        # Step 1: Check environment variables
        self.check_environment_variables()
        
        # Step 2: Get or verify API token
        self.verify_api_token()
        
        # Step 3: Test API authentication
        self.test_api_authentication()
        
        # Step 4: Test Git authentication methods
        self.test_git_authentication()
        
        # Step 5: Provide recommendations
        self.provide_recommendations()
    
    def check_environment_variables(self):
        """Check for Bitbucket environment variables"""
        print("1. Checking environment variables...")
        print("-" * 40)
        
        env_vars = {
            'BITBUCKET_API_TOKEN': os.getenv('BITBUCKET_API_TOKEN'),
            'BITBUCKET_USERNAME': os.getenv('BITBUCKET_USERNAME'),
            'BITBUCKET_APP_PASSWORD': os.getenv('BITBUCKET_APP_PASSWORD')
        }
        
        found_any = False
        for var, value in env_vars.items():
            if value:
                print(f"✓ {var} is set (length: {len(value)})")
                found_any = True
            else:
                print(f"✗ {var} is not set")
        
        if not found_any:
            print("\n⚠ No Bitbucket credentials found in environment variables")
        print()
    
    def verify_api_token(self):
        """Get or verify API token"""
        print("2. API Token Setup...")
        print("-" * 40)
        
        self.api_token = os.getenv('BITBUCKET_API_TOKEN')
        
        if self.api_token:
            print("✓ Found API token in environment")
        else:
            print("No API token found in environment.")
            print("\nTo create a Bitbucket API token:")
            print("1. Log into Bitbucket")
            print("2. Click your avatar → Personal settings")
            print("3. Click 'Access tokens' in the left sidebar")
            print("4. Click 'Create a token'")
            print("5. Name: 'Git Clone Access' (or any name)")
            print("6. Permissions needed:")
            print("   - Repository: Read")
            print("   - Repository: Write (if you need to push)")
            print("7. Click 'Create' and copy the token")
            print()
            
            self.api_token = getpass("Enter your Bitbucket API token: ").strip()
        
        print()
    
    def test_api_authentication(self):
        """Test API authentication"""
        print("3. Testing API Authentication...")
        print("-" * 40)
        
        if not self.api_token:
            print("✗ No API token available to test")
            return
        
        # Test API access
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json"
        }
        
        # Try to access user info
        response = requests.get(
            "https://api.bitbucket.org/2.0/user",
            headers=headers
        )
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"✓ API authentication successful!")
            print(f"  - Username: {user_info.get('username', 'N/A')}")
            print(f"  - Display name: {user_info.get('display_name', 'N/A')}")
            
            # Check repository access
            self.check_repository_access()
        else:
            print(f"✗ API authentication failed!")
            print(f"  - Status code: {response.status_code}")
            print(f"  - Error: {response.text[:200]}")
        
        print()
    
    def check_repository_access(self):
        """Check access to specific repository"""
        print("\n  Checking repository access...")
        
        # Extract workspace and repo from URL
        # https://bitbucket.org/f-secure/sbp_api.git
        workspace = "f-secure"
        repo_slug = "sbp_api"
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json"
        }
        
        response = requests.get(
            f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}",
            headers=headers
        )
        
        if response.status_code == 200:
            repo_info = response.json()
            print(f"  ✓ Repository access confirmed!")
            print(f"    - Name: {repo_info.get('name', 'N/A')}")
            print(f"    - Private: {repo_info.get('is_private', 'N/A')}")
        else:
            print(f"  ✗ Cannot access repository")
            print(f"    - Status: {response.status_code}")
            if response.status_code == 404:
                print("    - Repository not found or no access")
            elif response.status_code == 403:
                print("    - Access forbidden - check token permissions")
    
    def test_git_authentication(self):
        """Test different Git authentication methods"""
        print("4. Testing Git Clone Authentication Methods...")
        print("-" * 40)
        
        if not self.api_token:
            print("✗ No API token available to test")
            return
        
        test_dir = tempfile.mkdtemp()
        
        try:
            # Method 1: Using x-token-auth
            print("\nMethod 1: x-token-auth format...")
            url1 = f"https://x-token-auth:{self.api_token}@bitbucket.org/f-secure/sbp_api.git"
            result1 = self.test_clone(url1, test_dir, "method1")
            
            # Method 2: Using token as username
            print("\nMethod 2: Token as username...")
            url2 = f"https://{self.api_token}@bitbucket.org/f-secure/sbp_api.git"
            result2 = self.test_clone(url2, test_dir, "method2")
            
            # Method 3: Using git credential helper
            print("\nMethod 3: Git credential helper...")
            self.test_credential_helper()
            
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
        
        print()
    
    def test_clone(self, url, base_dir, method_name):
        """Test a specific clone method"""
        target_dir = Path(base_dir) / method_name
        
        try:
            # Set up environment
            env = os.environ.copy()
            env['GIT_HTTP_LOW_SPEED_LIMIT'] = '1000'
            env['GIT_HTTP_LOW_SPEED_TIME'] = '30'
            
            repo = git.Repo.clone_from(
                url,
                target_dir,
                branch="main",
                depth=1,
                env=env
            )
            
            print(f"  ✓ Clone successful!")
            print(f"    - Commit: {repo.head.commit.hexsha[:8]}")
            
            # Clean up
            shutil.rmtree(target_dir, ignore_errors=True)
            return True
            
        except git.exc.GitCommandError as e:
            print(f"  ✗ Clone failed: {str(e)[:100]}")
            return False
    
    def test_credential_helper(self):
        """Test git credential helper setup"""
        try:
            # Check current credential helper
            result = subprocess.run(
                ['git', 'config', '--global', 'credential.helper'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  Current credential helper: {result.stdout.strip()}")
            else:
                print("  No credential helper configured")
            
            # Provide setup instructions
            print("\n  To set up credential storage:")
            print("  Windows: git config --global credential.helper manager")
            print("  Mac: git config --global credential.helper osxkeychain")
            print("  Linux: git config --global credential.helper store")
            
        except Exception as e:
            print(f"  Could not check credential helper: {e}")
    
    def provide_recommendations(self):
        """Provide recommendations based on diagnostics"""
        print("5. Recommendations")
        print("-" * 40)
        
        print("\nRecommended setup:")
        print("1. Set environment variable:")
        print(f"   export BITBUCKET_API_TOKEN='your_token'")
        print("   (or on Windows: set BITBUCKET_API_TOKEN=your_token)")
        print()
        print("2. Use the clone tool:")
        print("   tool = EnhancedBitbucketCloneTool()")
        print("   result = tool._run('https://bitbucket.org/f-secure/sbp_api.git')")
        print()
        print("3. If issues persist, try:")
        print("   - Verify token has 'Repository: Read' permission")
        print("   - Check if you have access to the repository")
        print("   - Try cloning manually with:")
        print(f"     git clone https://x-token-auth:YOUR_TOKEN@bitbucket.org/f-secure/sbp_api.git")
        print()
        
    def test_with_tool(self):
        """Test using the actual EnhancedBitbucketCloneTool"""
        print("\n6. Testing with EnhancedBitbucketCloneTool...")
        print("-" * 40)
        
        if not self.api_token:
            print("✗ No API token available")
            return
        
        test_dir = tempfile.mkdtemp()
        
        try:
            # Create tool with token
            tool = EnhancedBitbucketCloneTool(
                workspace_dir=test_dir,
                api_token=self.api_token
            )
            
            print("Testing clone with the tool...")
            result_json = tool._run(
                "https://bitbucket.org/f-secure/sbp_api.git",
                branch="main"
            )
            
            import json
            result = json.loads(result_json)
            
            if result['success']:
                print("✓ Tool clone successful!")
                print(f"  - Files: {result['files_count']}")
                print(f"  - Commit: {result['short_commit_hash']}")
            else:
                print(f"✗ Tool clone failed: {result['error']}")
                
        except Exception as e:
            print(f"✗ Tool test failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    diagnostic = BitbucketAuthDiagnostic()
    diagnostic.run_diagnostics()
    diagnostic.test_with_tool()