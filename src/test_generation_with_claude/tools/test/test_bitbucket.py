"""
Real integration test script for EnhancedBitbucketCloneTool
Tests with actual Bitbucket repositories
"""

import json
import shutil
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
from getpass import getpass

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Import the tool class (adjust the import path as needed)
from bitbucket_tools import EnhancedBitbucketCloneTool, EnhancedBitbucketPRTool, BitbucketAnalyzerTool



class BitbucketIntegrationTest:
    """Integration tests for Bitbucket tools with real repositories"""
    
    def __init__(self):
        self.test_dir = tempfile.mkdtemp()
        self.results = []
        
    def cleanup(self):
        """Clean up test directories"""
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
            print("✓ Cleanup completed")
        except Exception as e:
            print(f"⚠ Cleanup warning: {e}")
    
    def print_header(self, title):
        """Print a formatted header"""
        print("\n" + "="*60)
        print(title)
        print("="*60 + "\n")
        
    def test_public_repo(self):
        """Test cloning a public repository (no auth required)"""
        self.print_header("Test 1: Public Repository Clone")
        
        # Use a public Bitbucket repository for testing
        public_repo = "https://bitbucket.org/atlassian/python-bitbucket.git"
        
        try:
            # Create tool without credentials
            tool = EnhancedBitbucketCloneTool(workspace_dir=self.test_dir)
            
            print(f"Cloning public repository: {public_repo}")
            start_time = datetime.now()
            
            # Clone without authentication
            result_json = tool._run(public_repo, branch="master", use_auth=False)
            result = json.loads(result_json)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if result['success']:
                print(f"✓ Public repo clone successful in {elapsed:.2f} seconds!")
                print(f"  - Repository: {result['repo_url']}")
                print(f"  - Branch: {result['branch']}")
                print(f"  - Commit: {result['short_commit_hash']}")
                print(f"  - Files: {result['files_count']}")
                print(f"  - Has tests: {result['has_tests']}")
                
                # Show file type distribution
                if result.get('file_types'):
                    print("  - Top file types:")
                    sorted_types = sorted(result['file_types'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]
                    for ext, count in sorted_types:
                        print(f"    • {ext}: {count} files")
                
                self.results.append(("Public repo clone", True, None))
            else:
                print(f"✗ Public repo clone failed: {result['error']}")
                self.results.append(("Public repo clone", False, result['error']))
                
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            self.results.append(("Public repo clone", False, str(e)))
    
    def test_private_repo_with_token(self, repo_url=None, branch="main"):
        """Test cloning a private repository with API token"""
        self.print_header("Test 2: Private Repository Clone with API Token")
        
        # Get API token
        api_token = os.getenv('BITBUCKET_API_TOKEN')
        if not api_token:
            print("No BITBUCKET_API_TOKEN found in environment.")
            api_token = getpass("Enter your Bitbucket API token: ").strip()
        
        if not api_token:
            print("⚠ Skipping private repo test - no API token provided")
            self.results.append(("Private repo clone", False, "No API token"))
            return
            
        # Get repository URL if not provided
        if not repo_url:
            print("\nEnter a private Bitbucket repository URL to test")
            print("Format: https://bitbucket.org/workspace/repo.git")
            repo_url = input("Repository URL (or press Enter to skip): ").strip()
            
        if not repo_url:
            print("⚠ Skipping private repo test - no repository URL provided")
            self.results.append(("Private repo clone", False, "No repo URL"))
            return
        
        try:
            # Create tool with API token
            tool = EnhancedBitbucketCloneTool(
                workspace_dir=self.test_dir,
                api_token=api_token
            )
            
            print(f"\nCloning private repository: {repo_url}")
            print(f"Branch: {branch}")
            start_time = datetime.now()
            
            # Clone with authentication
            result_json = tool._run(repo_url, branch=branch)
            result = json.loads(result_json)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if result['success']:
                print(f"✓ Private repo clone successful in {elapsed:.2f} seconds!")
                print(f"  - Repository: {result['repo_url']}")
                print(f"  - Branch: {result['branch']}")
                print(f"  - Commit: {result['short_commit_hash']}")
                print(f"  - Message: {result['commit_message'][:60]}...")
                print(f"  - Author: {result['author']}")
                print(f"  - Files: {result['files_count']}")
                print(f"  - Project type: {result.get('project_type', 'unknown')}")
                print(f"  - Framework: {result.get('framework', 'unknown')}")
                print(f"  - Has tests: {result['has_tests']}")
                
                if result['has_tests'] and result['test_files']:
                    print(f"  - Test files found: {len(result['test_files'])}")
                    for test_file in result['test_files'][:5]:
                        print(f"    • {test_file}")
                    if len(result['test_files']) > 5:
                        print(f"    ... and {len(result['test_files']) - 5} more")
                
                self.results.append(("Private repo clone", True, None))
                
                # Test cache functionality
                self.test_cache_performance(tool, repo_url, branch)
                
                # Test repository analysis
                self.test_repository_analysis(result['repo_path'])

                # Test 3: Error handling
                self.test_error_handling()
                
                # Print summary
                self.print_summary()
                
            else:
                print(f"✗ Private repo clone failed: {result['error']}")
                self.results.append(("Private repo clone", False, result['error']))
                
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            self.results.append(("Private repo clone", False, str(e)))
    
    def test_cache_performance(self, tool, repo_url, branch):
        """Test cache functionality"""
        self.print_header("Test 3: Cache Performance")
        
        try:
            # First clone (should be cached from previous test)
            start_time = datetime.now()
            result_json = tool._run(repo_url, branch=branch)
            cache_time = (datetime.now() - start_time).total_seconds()
            
            print(f"✓ Cached clone completed in {cache_time:.2f} seconds")
            print("  - This should be much faster than the initial clone")
            
            # Get cache statistics
            stats = tool.get_cache_stats()
            print(f"  - Cache size: {stats['cache_size']}")
            print(f"  - Cached repos: {len(stats['cached_repos'])}")
            
            self.results.append(("Cache test", True, None))
            
        except Exception as e:
            print(f"✗ Cache test failed: {e}")
            self.results.append(("Cache test", False, str(e)))
    
    def test_repository_analysis(self, repo_path):
        """Test repository analysis tool"""
        self.print_header("Test 4: Repository Analysis")
        
        try:
            analyzer = BitbucketAnalyzerTool()
            result = analyzer._run(repo_path)
            
            if result['success']:
                print("✓ Repository analysis successful!")
                print(f"  - Total files: {result['total_files']}")
                print(f"  - Python files: {result['python_files']['count']}")
                print(f"  - Test files: {result['test_files']['count']}")
                print(f"  - Config files: {result['config_files']['count']}")
                print(f"  - Documentation: {result['documentation']['count']}")
                print(f"  - Directories: {result['directories']['count']}")
                
                self.results.append(("Repository analysis", True, None))
            else:
                print(f"✗ Analysis failed: {result['error']}")
                self.results.append(("Repository analysis", False, result['error']))
                
        except Exception as e:
            print(f"✗ Analysis test failed: {e}")
            self.results.append(("Repository analysis", False, str(e)))
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        self.print_header("Test 5: Error Handling")
        
        tool = EnhancedBitbucketCloneTool(workspace_dir=self.test_dir)
        
        # Test 1: Invalid repository URL
        print("Testing invalid repository URL...")
        result_json = tool._run("https://bitbucket.org/invalid/does-not-exist.git")
        result = json.loads(result_json)
        
        if not result['success']:
            print("✓ Correctly handled non-existent repository")
            self.results.append(("Invalid repo handling", True, None))
        else:
            print("✗ Failed to detect invalid repository")
            self.results.append(("Invalid repo handling", False, "Should have failed"))
        
        # Test 2: Invalid branch
        print("\nTesting invalid branch...")
        result_json = tool._run(
            "https://bitbucket.org/atlassian/python-bitbucket.git",
            branch="invalid-branch-name"
        )
        result = json.loads(result_json)
        
        if not result['success']:
            print("✓ Correctly handled invalid branch")
            self.results.append(("Invalid branch handling", True, None))
        else:
            print("✗ Failed to detect invalid branch")
            self.results.append(("Invalid branch handling", False, "Should have failed"))
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("\n" + "="*60)
        print("Bitbucket Clone Tool Integration Tests")
        print("="*60)
        
        try:
            # Test 1: Public repository
            self.test_public_repo()
            
            # Test 2: Private repository (if credentials available)
            self.test_private_repo_with_token()
            
            # Test 3: Error handling
            self.test_error_handling()
            
            # Print summary
            self.print_summary()
            
        finally:
            self.cleanup()
    
    def print_summary(self):
        """Print test results summary"""
        self.print_header("Test Results Summary")
        
        passed = sum(1 for _, success, _ in self.results if success)
        failed = len(self.results) - passed
        
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print()
        
        for test_name, success, error in self.results:
            status = "✓" if success else "✗"
            print(f"{status} {test_name}")
            if error:
                print(f"  Error: {error}")
        
        print("\n" + ("="*60))
        if failed == 0:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {failed} test(s) failed")


def main():
    """Main test runner"""
    # Check if running in CI or interactive mode
    ci_mode = os.getenv('CI', '').lower() == 'true'
    
    if not ci_mode:
        print("Bitbucket Clone Tool - Real Integration Tests")
        print("This will test with actual Bitbucket repositories")
        print("\nOptions:")
        print("1. Run all tests")
        print("2. Test public repository only")
        print("3. Test private repository only")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '4':
            print("Exiting...")
            return
            
        test_runner = BitbucketIntegrationTest()
        
        if choice == '1':
            test_runner.run_all_tests()
        elif choice == '2':
            test_runner.test_public_repo()
            test_runner.cleanup()
        elif choice == '3':
            test_runner.test_private_repo_with_token(
                repo_url="https://bitbucket.org/alexleong12/pytest_backend_template.git",
                branch="main"
            )
            test_runner.cleanup()
        else:
            print("Invalid option")
    else:
        # CI mode - run all tests
        test_runner = BitbucketIntegrationTest()
        test_runner.run_all_tests()


if __name__ == "__main__":
    main()