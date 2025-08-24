#!/usr/bin/env python3
"""
Test script for OpenAPISpecReaderTool
Tests reading of OpenAPI/Swagger specifications in both JSON and YAML formats
Supports both local files and online specifications
"""

import os
import json
import tempfile
import sys
from pathlib import Path
import argparse

# Add the current directory to the Python path to enable imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from api_analyzer import OpenAPISpecReaderTool


def create_sample_openapi_json():
    """Create a sample OpenAPI 3.0 specification in JSON format"""
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Sample Pet Store API",
            "version": "1.0.0",
            "description": "A sample API for testing OpenAPISpecReaderTool"
        },
        "servers": [
            {
                "url": "https://api.example.com/v1",
                "description": "Production server"
            }
        ],
        "paths": {
            "/pets": {
                "get": {
                    "summary": "List all pets",
                    "operationId": "listPets",
                    "tags": ["pets"],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "How many items to return at one time",
                            "required": False,
                            "schema": {
                                "type": "integer",
                                "format": "int32"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "A paged array of pets",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Pets"
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "summary": "Create a pet",
                    "operationId": "createPets",
                    "tags": ["pets"],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Pet"
                                }
                            }
                        },
                        "required": True
                    },
                    "responses": {
                        "201": {
                            "description": "Null response"
                        }
                    }
                }
            },
            "/pets/{petId}": {
                "get": {
                    "summary": "Info for a specific pet",
                    "operationId": "showPetById",
                    "tags": ["pets"],
                    "parameters": [
                        {
                            "name": "petId",
                            "in": "path",
                            "required": True,
                            "description": "The id of the pet to retrieve",
                            "schema": {
                                "type": "string"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Expected response to a valid request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Pet"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "Pet": {
                    "type": "object",
                    "required": ["id", "name"],
                    "properties": {
                        "id": {
                            "type": "integer",
                            "format": "int64"
                        },
                        "name": {
                            "type": "string"
                        },
                        "tag": {
                            "type": "string"
                        }
                    }
                },
                "Pets": {
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/Pet"
                    }
                }
            }
        }
    }
    return openapi_spec


def create_sample_openapi_yaml():
    """Create a sample OpenAPI specification in YAML format"""
    yaml_content = """openapi: 3.0.0
info:
  title: Sample Order API
  version: 1.0.0
  description: API for managing orders
paths:
  /orders:
    get:
      summary: Get all orders
      operationId: getOrders
      responses:
        '200':
          description: List of orders
    post:
      summary: Create new order
      operationId: createOrder
      responses:
        '201':
          description: Order created
  /orders/{orderId}:
    get:
      summary: Get order by ID
      operationId: getOrderById
      parameters:
        - name: orderId
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Order details
    patch:
      summary: Update order
      operationId: updateOrder
      parameters:
        - name: orderId
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Order updated
"""
    return yaml_content


def print_result(result_str):
    """Pretty print the result from OpenAPISpecReaderTool"""
    try:
        # Parse the JSON result
        result_data = json.loads(result_str)
        
        # Check if it's an error response
        if "error" in result_data:
            print(f"❌ Error: {result_data['error']}")
            if 'metrics' in result_data:
                metrics = result_data['metrics']
                print(f"   Execution time: {metrics['total_execution_time']:.3f}s")
            print("-" * 50)
            return
        
        # For successful results
        print("✅ Success! OpenAPI spec loaded")
        print(f"   Format: JSON")
        print(f"   OpenAPI Version: {result_data.get('openapi', result_data.get('swagger', 'Unknown'))}")
        
        if 'info' in result_data:
            print(f"   API Title: {result_data['info'].get('title', 'N/A')}")
            print(f"   API Version: {result_data['info'].get('version', 'N/A')}")
            if 'description' in result_data['info']:
                desc = result_data['info']['description'][:100]
                if len(result_data['info']['description']) > 100:
                    desc += "..."
                print(f"   Description: {desc}")
        
        if 'servers' in result_data:
            print(f"   Servers: {len(result_data['servers'])}")
            for server in result_data['servers'][:2]:
                print(f"     - {server.get('url', 'N/A')}")
        
        if 'paths' in result_data:
            print(f"   Number of paths: {len(result_data['paths'])}")
            # Show first few paths
            paths = list(result_data['paths'].keys())[:5]
            if paths:
                print(f"   Sample paths:")
                for path in paths:
                    methods = list(result_data['paths'][path].keys())
                    methods = [m.upper() for m in methods if m in ['get', 'post', 'put', 'delete', 'patch']]
                    print(f"     - {path} [{', '.join(methods)}]")
                if len(result_data['paths']) > 5:
                    print(f"     ... and {len(result_data['paths']) - 5} more")
        
        print(f"   Content length: {len(result_str)} characters")
        print("-" * 50)
        
    except Exception as e:
        print(f"Failed to process result: {str(e)}")
        print(f"Raw result: {result_str[:200]}...")


def test_openapi_reader():
    """Test the OpenAPISpecReaderTool with various scenarios"""
    
    # Initialize the tool
    reader = OpenAPISpecReaderTool()
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        print("=== Testing OpenAPISpecReaderTool ===\n")
        
        # Test 1: Read JSON spec
        print("Test 1: Reading JSON OpenAPI spec")
        json_spec_path = os.path.join(temp_dir, "openapi.json")
        with open(json_spec_path, 'w') as f:
            json.dump(create_sample_openapi_json(), f, indent=2)
        
        result = reader._run(json_spec_path)
        print_result(result)
        
        # Test 2: Read YAML spec (converts to JSON)
        print("\nTest 2: Reading YAML OpenAPI spec (auto-converts to JSON)")
        yaml_spec_path = os.path.join(temp_dir, "openapi.yaml")
        with open(yaml_spec_path, 'w') as f:
            f.write(create_sample_openapi_yaml())
        
        result = reader._run(yaml_spec_path)
        print_result(result)
        
        # Test 3: Invalid file path
        print("\nTest 3: Testing with non-existent file")
        result = reader._run("/non/existent/path/openapi.json")
        print_result(result)
        
        # Test 4: Empty JSON file
        print("\nTest 4: Testing with empty JSON file")
        empty_json_path = os.path.join(temp_dir, "empty.json")
        with open(empty_json_path, 'w') as f:
            f.write("{}")
        
        result = reader._run(empty_json_path)
        print_result(result)
        
        # Test 5: Malformed JSON
        print("\nTest 5: Testing with malformed JSON")
        bad_json_path = os.path.join(temp_dir, "bad.json")
        with open(bad_json_path, 'w') as f:
            f.write('{"invalid": json content}')
        
        result = reader._run(bad_json_path)
        print_result(result)
        
        # Test 6: Complex nested spec
        print("\nTest 6: Testing with complex nested structure")
        complex_spec = create_sample_openapi_json()
        complex_spec["paths"]["/nested/{id}/items"] = {
            "get": {
                "summary": "Get nested items",
                "parameters": [
                    {"name": "id", "in": "path", "required": True},
                    {"name": "filter", "in": "query", "required": False}
                ],
                "responses": {"200": {"description": "Success"}}
            }
        }
        complex_spec_path = os.path.join(temp_dir, "complex.json")
        with open(complex_spec_path, 'w') as f:
            json.dump(complex_spec, f, indent=2)
        
        result = reader._run(complex_spec_path)
        print_result(result)


def test_online_specs():
    """Test reading online OpenAPI specifications"""
    reader = OpenAPISpecReaderTool()
    
    print("\n=== Testing Online OpenAPI Specifications ===\n")
    
    # List of public OpenAPI specs for testing
    online_specs = [
        {
            "name": "Petstore API (Swagger 2.0)",
            "url": "https://petstore.swagger.io/v2/swagger.json"
        },
        {
            "name": "GitHub API",
            "url": "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json"
        }
    ]
    
    for spec_info in online_specs:
        print(f"\nTesting: {spec_info['name']}")
        print(f"URL: {spec_info['url']}")
        try:
            result = reader._run(spec_info['url'])
            print_result(result)
            
        except Exception as e:
            print(f"❌ Failed to read: {str(e)}")
            print("-" * 50)


def read_custom_spec(spec_path):
    """Read a custom OpenAPI specification from file path or URL"""
    reader = OpenAPISpecReaderTool()
    
    print(f"\n=== Reading Custom Specification ===")
    print(f"Path/URL: {spec_path}")
    print(f"Output format: JSON\n")
    
    try:
        result = reader._run(spec_path)
        print_result(result)
        
        # Save the result to a JSON file
        output_file = "openapi_spec_output.json"
        
        with open(output_file, 'w') as f:
            f.write(result)
        
        print(f"\n✅ Spec content saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Failed to read specification: {str(e)}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test OpenAPISpecReaderTool with local files or online specifications"
    )
    parser.add_argument(
        "spec",
        nargs="?",
        help="Path to local OpenAPI spec file or URL to online spec"
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Run all built-in tests"
    )
    parser.add_argument(
        "--test-online",
        action="store_true",
        help="Test with known online OpenAPI specifications"
    )
    
    args = parser.parse_args()
    
    # Check if PyYAML is installed
    try:
        import yaml
        print("✓ PyYAML is installed - YAML files can be read\n")
    except ImportError:
        print("⚠ PyYAML is not installed - YAML file reading will fail")
        print("  Install with: pip install PyYAML\n")
    
    # Check if requests is installed (needed for online specs)
    try:
        import requests
        print("✓ requests is installed - Online spec reading will work\n")
    except ImportError:
        print("⚠ requests is not installed - Online spec reading will fail")
        print("  Install with: pip install requests\n")
    
    if args.spec:
        # Read custom specification
        read_custom_spec(args.spec)
    elif args.test_all:
        # Run all tests
        test_openapi_reader()
        test_online_specs()
    elif args.test_online:
        # Test only online specs
        test_online_specs()
    else:
        # Default: run built-in tests
        test_openapi_reader()
        print("\nTip: Use --help to see more options")
        print("Examples:")
        print("  python test_openapi_reader.py path/to/openapi.json")
        print("  python test_openapi_reader.py path/to/openapi.yaml")
        print("  python test_openapi_reader.py https://api.example.com/openapi.json")
        print("  python test_openapi_reader.py --test-online")
        print("  python test_openapi_reader.py --test-all")


if __name__ == "__main__":
    main()