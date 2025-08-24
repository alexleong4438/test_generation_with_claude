"""
Tools for Backend API Test Generation with Bitbucket Integration
"""

# API analysis tools
from .api_analyzer import (
    OpenAPISpecReaderTool,
    FastAPIAnalyzerTool,
)

# Xray integration tools
from .xray_tools import (
    EnhancedXrayAPITool,
)

# Bitbucket integration tools
from .bitbucket_tools import (
    EnhancedBitbucketCloneTool,
    EnhancedBitbucketPRTool,
    BitbucketAnalyzerTool,
)

# Test generation tools
from .test_generator import (
    PytestGeneratorTool,
    HTTPXTemplateTool,
    FixtureGeneratorTool,
)

# Test analysis tools
from .test_analysis_tools import (
    TestComparisonTool,
    TestCoverageAnalyzerTool,
    GapAnalyzerTool,
)

from .local_file_reader import (
    LocalFileReaderTool,
    TemplateFileReaderTool
)   

# Placeholder tools for backward compatibility
class ReadmeAnalyzerTool:
    """Placeholder for README analysis tool"""
    pass

class FilePatternAnalyzerTool:
    """Placeholder for file pattern analysis tool"""
    pass

class TestDuplicationDetectorTool:
    """Placeholder for test duplication detection tool"""
    pass

class TestScenarioDesignerTool:
    """Placeholder for test scenario design tool"""
    pass

class TestOrganizationTool:
    """Placeholder for test organization tool"""
    pass

class CoverageAnalyzerTool:
    """Placeholder for coverage analysis tool"""
    pass

class JiraAPITool:
    """Placeholder for Jira API tool"""
    pass

class RequirementParserTool:
    """Placeholder for requirement parser tool"""
    pass

class DjangoAnalyzerTool:
    """Placeholder for Django analyzer tool"""
    pass

class OpenAPIParserTool:
    """Placeholder for OpenAPI parser tool"""
    pass

class SchemaExtractorTool:
    """Placeholder for schema extractor tool"""
    pass

class TestModifierTool:
    """Placeholder for test modifier tool"""
    pass

class AssertionBuilderTool:
    """Placeholder for assertion builder tool"""
    pass

class SyntaxValidatorTool:
    """Placeholder for syntax validator tool"""
    pass

class DockerExecutorTool:
    """Placeholder for Docker executor tool"""
    pass

class MockValidatorTool:
    """Placeholder for mock validator tool"""
    pass

class RegressionCheckerTool:
    """Placeholder for regression checker tool"""
    pass

# LocalFileReaderTool and TemplateFileReaderTool are imported from local_file_reader module above

__all__ = [
    # API Analysis Tools
    "OpenAPISpecReaderTool",
    "FastAPIAnalyzerTool",
    
    # Xray Integration Tools
    "EnhancedXrayAPITool",
      # Bitbucket Integration Tools
    "EnhancedBitbucketCloneTool",
    "EnhancedBitbucketPRTool",
    "BitbucketAnalyzerTool",
      # Test Generation Tools
    "PytestGeneratorTool",
    "HTTPXTemplateTool",
    "FixtureGeneratorTool",
    
    # Test Analysis Tools
    "TestComparisonTool",
    "TestCoverageAnalyzerTool",
    "GapAnalyzerTool",
    
    # Placeholder Tools
    "ReadmeAnalyzerTool",
    "FilePatternAnalyzerTool",
    "TestDuplicationDetectorTool",
    "TestScenarioDesignerTool",
    "TestOrganizationTool",
    "CoverageAnalyzerTool",
    "JiraAPITool",
    "RequirementParserTool",
    "DjangoAnalyzerTool",
    "OpenAPIParserTool",
    "SchemaExtractorTool",
    "TestModifierTool",
    "AssertionBuilderTool",
    "SyntaxValidatorTool",
    "DockerExecutorTool",
    "MockValidatorTool",
    "RegressionCheckerTool",
    # Local File Reader Tools
    "LocalFileReaderTool",
    "TemplateFileReaderTool"
]
