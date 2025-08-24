#!/usr/bin/env python
"""
Backend API Test Generator with Bitbucket Integration
Generates or modifies API tests based on Jira requirements with enhanced Bitbucket repository analysis
"""

import os
import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from .crew import BackendAPITestCrew

# Load environment variables
load_dotenv()

def validate_inputs(args):
    """Validate required inputs for the enhanced workflow"""
    if not args.jira_key:
        raise ValueError("Jira key is required")
    
    if not args.bitbucket_repo:
        raise ValueError("Bitbucket repository URL is required")
    
    #if not os.path.exists(args.backend_path):
    #    raise ValueError(f"Backend path does not exist: {args.backend_path}")
    
    if args.framework not in ["fastapi", "django"]:
        raise ValueError("Framework must be 'fastapi' or 'django'")
    
    if args.action not in ["modify", "add", "auto"]:
        raise ValueError("Action must be 'modify', 'add', or 'auto'")
    
    # Validate Bitbucket credentials
    if not os.getenv("BITBUCKET_USERNAME") or not os.getenv("BITBUCKET_API_TOKEN"):
        raise ValueError("BITBUCKET_USERNAME and BITBUCKET_API_TOKEN environment variables are required")
    
    # Validate Jira credentials
    if not os.getenv("JIRA_BASE_URL") or not os.getenv("JIRA_EMAIL") or not os.getenv("JIRA_API_TOKEN"):
        raise ValueError("JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_TOKEN environment variables are required")

def setup_directories(args):
    """Create necessary directories for the workflow"""
    directories = [
        args.output_dir,
        args.workspace_dir,
        "output",
        f"{args.workspace_dir}/bitbucket_repo",
        f"{args.workspace_dir}/test_analysis",
        f"{args.workspace_dir}/generated_tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(
        description="Generate or modify API tests from Jira requirements with Bitbucket integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Enhanced Workflow Features:
                - Clone and analyze Bitbucket repositories
                - Compare existing tests with requirements
                - Identify test gaps and coverage issues
                - Generate or modify tests based on analysis
                - Validate generated tests
                - Create pull requests with changes

                Environment Variables Required:
                - JIRA_BASE_URL: Base URL of your Jira instance
                - JIRA_EMAIL: Your Jira email
                - JIRA_API_TOKEN: Jira API token
                - BITBUCKET_USERNAME: Bitbucket username
                - BITBUCKET_API_TOKEN: Bitbucket API token
        """
    )
    
    # Core arguments
    parser.add_argument(
        "--jira-key",
        required=True,
        help="Jira ticket key (e.g., PROJ-123)"
    )
    parser.add_argument(
        "--bitbucket-repo",
        required=True,
        help="Bitbucket repository URL for test cases (e.g., https://bitbucket.org/workspace/repo)"
    )
    parser.add_argument(
        "--bitbucket-branch",
        default="main",
        help="Git branch to work on (default: main)"
    )
    parser.add_argument(
        "--backend-path",
        default="./src",
        help="Path to backend code to analyze (default: ./src)"
    )
    
    # Framework and action options
    parser.add_argument(
        "--framework",
        choices=["fastapi", "django"],
        default="fastapi",
        help="Backend framework (default: fastapi)"
    )
    parser.add_argument(
        "--action",
        choices=["modify", "add", "auto"],
        default="auto",
        help="Test action: modify existing, add new, or auto-detect (default: auto)"
    )
    
    # Directory options
    parser.add_argument(
        "--output-dir",
        default="./tests",
        help="Output directory for tests (default: ./tests)"
    )
    parser.add_argument(
        "--workspace-dir",
        default="./workspace",
        help="Workspace directory for temporary files (default: ./workspace)"
    )
    
    # Analysis options
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        default=80,
        help="Minimum coverage threshold percentage (default: 80)"
    )
    parser.add_argument(
        "--test-pattern",
        default="test_*.py",
        help="Pattern for test files to analyze (default: test_*.py)"
    )
    
    # Workflow control options
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning repository (use existing workspace)"
    )
    parser.add_argument(
        "--skip-pr",
        action="store_true",
        help="Skip creating pull request"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform analysis only, don't generate or modify tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )    
    args = parser.parse_args()
    
    try:
        validate_inputs(args)
        setup_directories(args)
    except ValueError as e:        
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Setup Error: {e}")
        sys.exit(1)
    
    # Prepare inputs for the enhanced crew workflow
    inputs = {
        # Jira configuration
        "jira_key": args.jira_key,
        "jira_url": os.getenv("JIRA_URL"),
        "jira_token": os.getenv("JIRA_TOKEN"),
        
        # Bitbucket configuration
        "bitbucket_repo": args.bitbucket_repo,
        "bitbucket_branch": args.bitbucket_branch,
        "bitbucket_username": os.getenv("BITBUCKET_USERNAME"),
        "bitbucket_app_password": os.getenv("BITBUCKET_APP_PASSWORD"),
        
        # Backend analysis paths
        "backend_paths": args.backend_path,
        "framework": args.framework,
        
        # Output and workspace configuration
        "output_dir": args.output_dir,
        "workspace_dir": args.workspace_dir,
        
        # Analysis parameters
        "coverage_threshold": args.coverage_threshold,
        "test_pattern": args.test_pattern,
        "action": args.action,
        
        # Workflow control
        "skip_clone": args.skip_clone,
        "skip_pr": args.skip_pr,
        "dry_run": args.dry_run,
        "verbose": args.verbose
    }
    
    print("[START] Starting API Test Generation Workflow")
    print("=" * 60)
    print(f"[INFO] Jira Ticket: {args.jira_key}")
    print(f"[LINK] Bitbucket Repository: {args.bitbucket_repo}")
    print(f"[BRANCH] Branch: {args.bitbucket_branch}")
    print(f"[FOLDER] Backend Path: {args.backend_path}")
    print(f"[TOOL] Framework: {args.framework}")
    print(f"[TARGET] Action: {args.action}")
    print(f"[STATS] Coverage Threshold: {args.coverage_threshold}%")
    print(f"[STORAGE] Workspace: {args.workspace_dir}")    
    print(f"[OUTPUT] Output: {args.output_dir}")
    if args.dry_run:
        print("[SEARCH] Mode: Dry Run (Analysis Only)")
    print("=" * 60)
    
    try:
        # Initialize and run the enhanced crew
        crew = BackendAPITestCrew()
        result = crew.crew().kickoff(inputs=inputs)
        
        # Save detailed results
        result_file = Path("output") / f"enhanced_result_{args.jira_key}.json"
        with open(result_file, "w") as f:
            if hasattr(result, 'dict'):
                json.dump(result.dict(), f, indent=2, default=str)
            else:
                json.dump(str(result), f, indent=2)
        
        print("\n" + "=" * 60)
        print("[OK] Enhanced Test Generation Workflow Completed!")
        print("=" * 60)
        print(f"[STATS] Detailed results saved to: {result_file}")
        print(f"🧪 Tests output directory: {args.output_dir}")
        print(f"[STORAGE] Workspace directory: {args.workspace_dir}")
        
        '''
        # Display workflow summary
        if hasattr(result, 'summary'):
            summary = result.summary
            print(f"\n[METRICS] Workflow Summary:")
            print(f"   • Repository Analysis: {'[OK] Completed' if summary.get('repo_analyzed') else '[ERROR] Failed'}")
            print(f"   • Test Gap Analysis: {'[OK] Completed' if summary.get('gaps_analyzed') else '[ERROR] Failed'}")
            print(f"   • Requirements Extraction: {'[OK] Completed' if summary.get('requirements_extracted') else '[ERROR] Failed'}")
            print(f"   • Test Comparison: {'[OK] Completed' if summary.get('tests_compared') else '[ERROR] Failed'}")
            print(f"   • Test Generation/Modification: {'[OK] Completed' if summary.get('tests_modified') else '[ERROR] Failed'}")
            print(f"   • Test Validation: {'[OK] Completed' if summary.get('tests_validated') else '[ERROR] Failed'}")
            if not args.skip_pr:
                print(f"   • Pull Request: {'[OK] Created' if summary.get('pr_created') else '[ERROR] Failed'}")
        
        # Display metrics if available
        if hasattr(result, 'metrics'):
            metrics = result.metrics
            print(f"\n[STATS] Quality Metrics:")
            if metrics.get('coverage_percentage'):
                print(f"   • Coverage: {metrics['coverage_percentage']}%")
            if metrics.get('tests_generated'):
                print(f"   • Tests Generated: {metrics['tests_generated']}")
            if metrics.get('tests_modified'):
                print(f"   • Tests Modified: {metrics['tests_modified']}")
            if metrics.get('test_gaps_found'):
                print(f"   • Test Gaps Found: {metrics['test_gaps_found']}")
            if metrics.get('requirements_covered'):
                print(f"   • Requirements Covered: {metrics['requirements_covered']}")
        '''
        
        print("\n[SUCCESS] Workflow completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Error during workflow execution: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        print("[ERROR] Full traceback:")
        traceback.print_exc()
        
        # Additional debugging information
        print(f"\n[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Python path: {sys.path[:3]}...")  # Show first 3 entries
        
        sys.exit(1)

if __name__ == "__main__":
    main()
