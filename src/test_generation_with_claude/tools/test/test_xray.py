"""
Real-world testing script for GET_TEST_CASE operation against actual Jira/Xray instance.
This script connects to your real Jira instance and performs various GET_TEST_CASE operations.
"""

import os
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, List

from anyio import Path

# Add the current directory to the Python path to enable imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the tool (adjust path as needed)
try:
    from xray_tools import EnhancedXrayAPITool, XrayOperation
except ImportError:
    print("Error: Could not import EnhancedXrayAPITool")
    print("Make sure the enhanced_xray_api_tool.py file is in the same directory or in PYTHONPATH")
    sys.exit(1)

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

class XrayGetTestCaseTester:
    """Test GET_TEST_CASE operation against real Jira/Xray instance"""
    
    def __init__(self):
        self.tool = None
        self.test_results = []
        
    def setup(self) -> bool:
        """Setup and validate environment"""
        print("🔧 Setting up Xray API Tool...\n")
        
        # Check required environment variables
        required_vars = {
            "JIRA_BASE_URL": "Your Jira instance URL (e.g., https://company.atlassian.net)",
            "JIRA_EMAIL": "Your Jira email address",
            "JIRA_API_TOKEN": "Your Jira API token (get from: https://id.atlassian.com/manage-profile/security/api-tokens)"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"   {var}: {description}")
        
        if missing_vars:
            print("❌ Missing required environment variables:\n")
            print("\n".join(missing_vars))
            print("\nSet them using:")
            print("export JIRA_BASE_URL='https://your-instance.atlassian.net'")
            print("export JIRA_EMAIL='your-email@example.com'")
            print("export JIRA_API_TOKEN='your-api-token'")
            return False
        
        # Display current configuration
        print("📋 Current Configuration:")
        print(f"   JIRA_BASE_URL: {os.getenv('JIRA_BASE_URL')}")
        print(f"   JIRA_EMAIL: {os.getenv('JIRA_EMAIL')}")
        print(f"   JIRA_API_TOKEN: {'*' * 10} (hidden)")
        
        # Optional Xray field configuration
        xray_fields = {
            "XRAY_STEPS_FIELD": os.getenv("XRAY_STEPS_FIELD", "customfield_10000"),
            "XRAY_TEST_TYPE_FIELD": os.getenv("XRAY_TEST_TYPE_FIELD", "customfield_10001"),
            "XRAY_PRECONDITIONS_FIELD": os.getenv("XRAY_PRECONDITIONS_FIELD", "customfield_10002"),
        }
        
        print("\n📋 Xray Custom Fields:")
        for field, value in xray_fields.items():
            print(f"   {field}: {value}")
        
        try:
            # Initialize the tool
            print("\n🚀 Initializing Xray API Tool...")
            self.tool = EnhancedXrayAPITool()
            print("✅ Tool initialized successfully!\n")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize tool: {str(e)}")
            return False
    
    def test_single_test_case(self, test_key: str) -> Dict[str, Any]:
        """Test getting a single test case with all details"""
        print(f"\n{'='*60}")
        print(f"📋 TEST 1: Get Single Test Case - {test_key}")
        print(f"{'='*60}")
        
        test_result = {
            "test_name": "Get Single Test Case",
            "test_key": test_key,
            "status": "FAILED",
            "duration": 0,
            "error": None,
            "data": None
        }
        
        try:
            start_time = time.time()
            
            # Get test case with all details
            print(f"🔍 Fetching test case {test_key} with all details...")
            result = self.tool._run(
                operation=XrayOperation.GET_TEST_CASE,
                test_case_key=test_key,
                include_steps=True,
                include_attachments=True,
                include_requirements=True,
                include_executions=True
            )
            
            duration = time.time() - start_time
            test_result["duration"] = duration
            
            # Parse the result
            data = json.loads(result)
            test_result["data"] = data
            test_result["status"] = "PASSED"
            
            # Display results
            print(f"\n✅ Successfully retrieved test case in {duration:.2f} seconds")
            print(f"\n📊 Test Case Details:")
            print(f"   Key: {data.get('key')}")
            print(f"   Summary: {data.get('summary')}")
            print(f"   Description: {data.get('description', 'Not specified')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Priority: {data.get('priority')}")
            print(f"   Test Type: {data.get('test_type', 'Not specified')}")
            print(f"   Reporter: {data.get('reporter')}")
            print(f"   Assignee: {data.get('assignee')}")
            print(f"   Created: {data.get('created')}")
            print(f"   Updated: {data.get('updated')}")
            
            # Description
            if data.get('description'):
                print(f"\n📝 Description:")
                print(f"   {data['description'][:200]}{'...' if len(data.get('description', '')) > 200 else ''}")
            
            # Preconditions
            if data.get('preconditions'):
                print(f"\n⚡ Preconditions:")
                print(f"   {data['preconditions'][:200]}{'...' if len(data.get('preconditions', '')) > 200 else ''}")
            
            # Labels
            if data.get('labels'):
                print(f"\n🏷️  Labels: {', '.join(data['labels'])}")
            
            # Components
            if data.get('components'):
                print(f"\n🧩 Components: {', '.join(data['components'])}")
            
            # Test Steps
            if 'steps' in data and data['steps']:
                print(f"\n📝 Test Steps: {len(data['steps'])} steps")
                for i, step in enumerate(data['steps'][:3], 1):  # Show first 3 steps
                    print(f"\n   Step {step.get('index', i)}:")
                    print(f"      Action: {step.get('action', 'N/A')}")
                    if step.get('data'):
                        print(f"      Data: {step['data']}")
                    if step.get('expected_result'):
                        print(f"      Expected: {step['expected_result']}")
                
                if len(data['steps']) > 3:
                    print(f"\n   ... and {len(data['steps']) - 3} more steps")
            else:
                print(f"\n📝 Test Steps: None defined")
            
            # Attachments
            if 'attachments' in data and data['attachments']:
                print(f"\n📎 Attachments: {len(data['attachments'])} files")
                for att in data['attachments'][:3]:  # Show first 3
                    print(f"   - {att.get('filename')} ({att.get('size', 0)} bytes, {att.get('mimeType', 'unknown')})")
            
            # Requirements
            if 'requirements' in data and data['requirements']:
                print(f"\n🎯 Linked Requirements: {len(data['requirements'])}")
                for req in data['requirements'][:3]:  # Show first 3
                    print(f"   - {req.get('key')}: {req.get('summary')} [{req.get('status')}]")
            
            # Executions
            if 'executions' in data and data['executions']:
                print(f"\n🏃 Test Executions: {len(data['executions'])}")
                for exec in data['executions'][:3]:  # Show first 3
                    print(f"   - {exec.get('key')}: {exec.get('summary')} [{exec.get('status')}]")
            
            # Save full JSON for reference
            filename = f"test_case_{test_key.replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 Full test case data saved to: {filename}")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"\n❌ Error: {str(e)}")
            
            # Try to provide more context for common errors
            if "404" in str(e) or "not found" in str(e).lower():
                print(f"   → Test case {test_key} does not exist or you don't have permission to view it")
            elif "401" in str(e):
                print("   → Authentication failed. Check your JIRA_EMAIL and JIRA_API_TOKEN")
            elif "403" in str(e):
                print("   → Permission denied. You don't have access to this test case")
            
        self.test_results.append(test_result)
        return test_result
    
    def test_minimal_fetch(self, test_key: str) -> Dict[str, Any]:
        """Test getting test case with minimal data"""
        print(f"\n{'='*60}")
        print(f"📋 TEST 2: Get Test Case (Minimal) - {test_key}")
        print(f"{'='*60}")
        
        test_result = {
            "test_name": "Get Test Case (Minimal)",
            "test_key": test_key,
            "status": "FAILED",
            "duration": 0,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            print(f"🔍 Fetching test case {test_key} with minimal details...")
            result = self.tool._run(
                operation=XrayOperation.GET_TEST_CASE,
                test_case_key=test_key,
                include_steps=False,
                include_attachments=False,
                include_requirements=False,
                include_executions=False
            )
            
            duration = time.time() - start_time
            test_result["duration"] = duration
            
            data = json.loads(result)
            test_result["status"] = "PASSED"
            
            print(f"\n✅ Retrieved minimal test case in {duration:.2f} seconds")
            print(f"   Key: {data.get('key')}")
            print(f"   Summary: {data.get('summary')}")
            print(f"   Status: {data.get('status')}")
            
            # Compare response sizes
            if hasattr(self, 'full_response_size'):                
                minimal_size = len(result)
                
                print(f"\n📊 Response Size Comparison:")
                print(f"   Full details: {self.full_response_size:,} bytes")
                print(f"   Minimal: {minimal_size:,} bytes")
                
                # Handle the case where full_response_size might be 0
                if self.full_response_size > 0:
                    reduction_percent = ((1 - minimal_size/self.full_response_size) * 100)
                    print(f"   Reduction: {reduction_percent:.1f}%")
                else:
                    print(f"   Reduction: Unable to calculate (full response size is 0)")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"\n❌ Error: {str(e)}")
            
        self.test_results.append(test_result)
        return test_result
    
    def test_bulk_fetch(self, test_keys: List[str]) -> Dict[str, Any]:
        """Test bulk fetching multiple test cases"""
        print(f"\n{'='*60}")
        print(f"📋 TEST 3: Bulk Fetch - {len(test_keys)} Test Cases")
        print(f"{'='*60}")
        
        test_result = {
            "test_name": "Bulk Fetch Test Cases",
            "test_keys": test_keys,
            "status": "FAILED",
            "duration": 0,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            print(f"🔍 Fetching {len(test_keys)} test cases: {', '.join(test_keys)}")
            result = self.tool._run(
                operation=XrayOperation.GET_TEST_CASE,
                test_case_keys=test_keys,
                include_steps=True,
                include_attachments=False,
                include_requirements=False,
                include_executions=False
            )
            
            duration = time.time() - start_time
            test_result["duration"] = duration
            
            data = json.loads(result)
            test_result["status"] = "PASSED"
            
            print(f"\n✅ Bulk fetch completed in {duration:.2f} seconds")
            print(f"\n📊 Summary:")
            print(f"   Total requested: {data['summary']['total_requested']}")
            print(f"   Successful: {data['summary']['successful']}")
            print(f"   Failed: {data['summary']['failed']}")
            print(f"   Average time per test case: {duration/len(test_keys):.2f} seconds")
            
            if data['test_cases']:
                print(f"\n✅ Successfully Retrieved:")
                for tc in data['test_cases']:
                    tc_data = json.loads(tc) if isinstance(tc, str) else tc
                    steps_count = len(tc_data.get('steps', []))
                    print(f"   - {tc_data['key']}: {tc_data['summary']}")
                    print(f"     Status: {tc_data['status']}, Steps: {steps_count}")
            
            if data['errors']:
                print(f"\n❌ Failed to Retrieve:")
                for error in data['errors']:
                    print(f"   - {error['key']}: {error['error']}")
            
            # Save bulk results
            filename = f"bulk_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 Bulk fetch results saved to: {filename}")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"\n❌ Error: {str(e)}")
            
        self.test_results.append(test_result)
        return test_result
    
    def test_cache_performance(self, test_key: str) -> Dict[str, Any]:
        """Test cache functionality and performance"""
        print(f"\n{'='*60}")
        print(f"📋 TEST 4: Cache Performance - {test_key}")
        print(f"{'='*60}")
        
        test_result = {
            "test_name": "Cache Performance",
            "test_key": test_key,
            "status": "FAILED",
            "error": None
        }
        
        try:
            # Clear cache first
            print("🧹 Clearing cache...")
            self.tool.clear_cache()
            
            # First fetch (no cache)
            print(f"\n🔍 First fetch (no cache)...")
            start_time = time.time()
            result1 = self.tool._run(
                operation=XrayOperation.GET_TEST_CASE,
                test_case_key=test_key,
                include_steps=True
            )
            first_duration = time.time() - start_time
            
            # Second fetch (should hit cache)
            start_time = time.time()
            result2 = self.tool._run(
                operation=XrayOperation.GET_TEST_CASE,
                test_case_key=test_key,
                include_steps=True
            )
            cached_duration = time.time() - start_time
            # Verify results are identical
            if result1 == result2:
                print("✅ Cache returned identical results")
            else:
                print("⚠️  Warning: Cached results differ from original")
              # Performance comparison
            print(f"\n📊 Performance Comparison:")
            print(f"   First fetch (API call): {first_duration:.5f} seconds")
            print(f"   Second fetch (cached): {cached_duration:.5f} seconds")
            
            # Handle the case where cached_duration is 0 (very fast cache)
            if cached_duration > 0:
                speed_improvement = first_duration / cached_duration
                print(f"   Speed improvement: {speed_improvement:.3f}x faster")
            else:
                print(f"   Speed improvement: ∞x faster (cached response was instantaneous)")
            
            print(f"   Time saved: {first_duration - cached_duration:.5f} seconds")
            
            # Cache stats
            cache_stats = self.tool.get_cache_stats()
            print(f"\n📊 Cache Statistics:")
            print(f"   Total entries: {cache_stats['total_entries']}")
            print(f"   Cache size: {cache_stats['cache_size_bytes']:,} bytes")
            for cache_type, count in cache_stats['entries_by_type'].items():
                print(f"   {cache_type}: {count} entries")
            
            test_result["status"] = "PASSED"
            test_result["first_duration"] = first_duration
            test_result["cached_duration"] = cached_duration
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"\n❌ Error: {str(e)}")
            
        self.test_results.append(test_result)
        return test_result
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid inputs"""
        print(f"\n{'='*60}")
        print(f"📋 TEST 5: Error Handling")
        print(f"{'='*60}")
        
        test_result = {
            "test_name": "Error Handling",
            "status": "PASSED",
            "errors_caught": []
        }
        
        # Test 1: Non-existent test case
        print("\n🔍 Testing non-existent test case...")
        try:
            self.tool._run(
                operation=XrayOperation.GET_TEST_CASE,
                test_case_key="INVALID-99999"
            )
            print("❌ Expected error was not raised")
            test_result["status"] = "FAILED"
        except Exception as e:
            print(f"✅ Correctly caught error: {type(e).__name__}: {str(e)}")
            test_result["errors_caught"].append("non-existent test case")
        
        # Test 2: Empty test key
        print("\n🔍 Testing empty test key...")
        try:
            self.tool._run(
                operation=XrayOperation.GET_TEST_CASE,
                test_case_key=""
            )
            print("❌ Expected error was not raised")
            test_result["status"] = "FAILED"
        except Exception as e:
            print(f"✅ Correctly caught error: {type(e).__name__}: {str(e)}")
            test_result["errors_caught"].append("empty test key")
        
        # Test 3: Invalid test key format
        print("\n🔍 Testing invalid test key format...")
        try:
            self.tool._run(
                operation=XrayOperation.GET_TEST_CASE,
                test_case_key="Not-A-Valid-Key!"
            )
            print("❌ Expected error was not raised")
            test_result["status"] = "FAILED"
        except Exception as e:
            print(f"✅ Correctly caught error: {type(e).__name__}: {str(e)}")
            test_result["errors_caught"].append("invalid key format")
        
        self.test_results.append(test_result)
        return test_result
    
    def generate_summary_report(self):
        """Generate a summary report of all tests"""
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY REPORT")
        print(f"{'='*60}")
        
        passed = sum(1 for t in self.test_results if t["status"] == "PASSED")
        failed = sum(1 for t in self.test_results if t["status"] == "FAILED")
        
        print(f"\n📈 Overall Results:")
        print(f"   Total Tests: {len(self.test_results)}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   Success Rate: {(passed/len(self.test_results)*100):.1f}%")
        
        print(f"\n📋 Test Details:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"\n   {status_icon} {result['test_name']}")
            
            if "test_key" in result:
                print(f"      Test Key: {result['test_key']}")
            elif "test_keys" in result:
                print(f"      Test Keys: {', '.join(result['test_keys'])}")
            
            if "duration" in result and result["duration"] > 0:
                print(f"      Duration: {result['duration']:.2f} seconds")
            
            if result.get("error"):
                print(f"      Error: {result['error']}")
        
        # Save summary report
        report_filename = f"test_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": len(self.test_results),
                    "passed": passed,
                    "failed": failed,
                    "success_rate": f"{(passed/len(self.test_results)*100):.1f}%"
                },
                "results": self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Full report saved to: {report_filename}")


def main():
    """Main test execution function"""
    print("🚀 Xray GET_TEST_CASE Real-World Testing")
    print("=" * 60)
    
    tester = XrayGetTestCaseTester()
    
    # Setup
    if not tester.setup():
        return
    
    # Get test inputs from user
    print("\n📝 Test Configuration")
    print("-" * 60)
    
    # Single test case
    test_key = input("\nEnter a test case key for single fetch (e.g., TEST-123): ").strip()
    if not test_key:
        print("⚠️  No test key provided. Using default: TEST-1")
        test_key = "TEST-1"
    
    # Multiple test cases for bulk fetch
    bulk_input = input("\nEnter multiple test case keys for bulk fetch, separated by comma (e.g., TEST-1,TEST-2,TEST-3): ").strip()
    if bulk_input:
        test_keys = [k.strip() for k in bulk_input.split(',') if k.strip()]
    else:
        print("⚠️  No bulk test keys provided. Using single test key for bulk test")
        test_keys = [test_key]
    
    # Run tests
    print(f"\n🏃 Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Single test case with all details
    tester.test_single_test_case(test_key)
    
    # Test 2: Minimal fetch
    tester.test_minimal_fetch(test_key)
    
    # Test 3: Bulk fetch
    if len(test_keys) > 1:
        tester.test_bulk_fetch(test_keys)
    else:
        print(f"\n⏭️  Skipping bulk fetch test (need multiple test keys)")
    
    # Test 4: Cache performance
    tester.test_cache_performance(test_key)
    
    # Test 5: Error handling
    tester.test_error_handling()
    
    # Generate summary report
    tester.generate_summary_report()
    
    print(f"\n✅ All tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Tips:")
    print("   - Check the generated JSON files for detailed test case data")
    print("   - Review any errors and ensure your test cases exist in Jira")
    print("   - Verify your custom field IDs match your Xray configuration")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()