from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from langchain_aws import BedrockLLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter

import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional

from openai import BaseModel

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

from tools import (
    # API Analysis Tools
    OpenAPISpecReaderTool, FastAPIAnalyzerTool,
    # Bitbucket Integration Tools
    EnhancedBitbucketCloneTool, EnhancedBitbucketPRTool, BitbucketAnalyzerTool,
    # Test Analysis Tools
    TestComparisonTool, TestCoverageAnalyzerTool, GapAnalyzerTool,
    # Test Generation Tools
    PytestGeneratorTool, HTTPXTemplateTool, FixtureGeneratorTool,
    # Xray Integration Tools
    EnhancedXrayAPITool,
    LocalFileReaderTool, TemplateFileReaderTool
    # Placeholder tools for backward compatibility
    #ReadmeAnalyzerTool, FilePatternAnalyzerTool, TestDuplicationDetectorTool,
    #TestScenarioDesignerTool, TestOrganizationTool, CoverageAnalyzerTool,
    #JiraAPITool, RequirementParserTool, DjangoAnalyzerTool, OpenAPIParserTool,
    #SchemaExtractorTool, TestModifierTool, AssertionBuilderTool, SyntaxValidatorTool,
    #DockerExecutorTool, MockValidatorTool, RegressionCheckerTool
)

# Load environment variables
load_dotenv()

class Coverage(BaseModel):
    number_of_test_files: int
    files: List[str]
    overall_coverage: str

class Structure(BaseModel):
    directory: str
    naming_convention: str
    uses_fixtures: bool
    uses_utilities: bool

class RepoAnalysisJson(BaseModel):
    framework: str
    test_location: str
    coverage: Coverage
    structure: Structure

# Output schema definitions
class TestScenarioJson(BaseModel):
    name: str
    method: str
    endpoint: str
    expected_status: Optional[int] = None
    expected_response: Optional[str] = None
    priority: Optional[str] = None
    execution_order: Optional[int] = None

class TestScenariosJson(BaseModel):
    scenarios: List[TestScenarioJson]


@CrewBase
class BackendAPITestCrew():
    """Enhanced Backend API Test Generation crew with Bitbucket integration"""
    
    agents: List[BaseAgent]
    tasks: List[Task]
    
    def __init__(self):
        """Initialize the crew"""
        super().__init__()
        self.llm = self._setup_llm()
        self._setup_tools()
        self._setup_workspace()
        # Initialize the tool usage count for Bedrock Knowledge Base Retriever Tool
        ConsoleFormatter.tool_usage_counts['Bedrock Knowledge Base Retriever Tool'] = 0
    
    def _setup_llm(self):
        """Configure AWS Bedrock LLM"""        # Get model from environment variable or use default
        model = os.getenv('MODEL', 'bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0')
        
        return LLM(
            model=model,
            temperature=0.1,  # Low temperature for consistent code generation
            max_tokens=5000
        )

    def _setup_workspace(self):
        """Setup workspace directories"""
        workspace_dirs = [
            "output",
            "workspace/repos",
            "tests/api",
            "tests/integration"
        ]
        for dir_path in workspace_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_tools(self):
        """Initialize tools for agents"""
        
        self.OpenAPISpecReaderTool = OpenAPISpecReaderTool()
        self.FastAPIAnalyzerTool = FastAPIAnalyzerTool()    
        self.EnhancedBitbucketCloneTool = EnhancedBitbucketCloneTool(workspace_dir="workspace/repos")
        self.EnhancedBitbucketPRTool = EnhancedBitbucketPRTool()
        self.BitbucketAnalyzerTool = BitbucketAnalyzerTool()
        self.TestComparisonTool = TestComparisonTool()
        self.TestCoverageAnalyzerTool = TestCoverageAnalyzerTool()
        self.GapAnalyzerTool = GapAnalyzerTool()
        self.PytestGeneratorTool = PytestGeneratorTool()
        self.HTTPXTemplateTool = HTTPXTemplateTool()
        self.FixtureGeneratorTool = FixtureGeneratorTool()
        self.EnhancedXrayAPITool = EnhancedXrayAPITool()

        self.LocalFileReaderTool = LocalFileReaderTool()
        self.TemplateFileReaderTool = TemplateFileReaderTool()

        """
        return {
            # API Analysis Tools
            "OpenAPISpecReaderTool": OpenAPISpecReaderTool(),
            "FastAPIAnalyzerTool": FastAPIAnalyzerTool(),
            #"DjangoAnalyzerTool": lambda: DjangoAnalyzerTool(),
            #"OpenAPIParserTool": lambda: OpenAPIParserTool(),
            #"SchemaExtractorTool": lambda: SchemaExtractorTool(),
            
            # Bitbucket Integration Tools
            "EnhancedBitbucketCloneTool": EnhancedBitbucketCloneTool(
                workspace_dir="workspace/repos"
            ),
            "EnhancedBitbucketPRTool": EnhancedBitbucketPRTool(),
            "BitbucketAnalyzerTool": BitbucketAnalyzerTool(),

            # Test Analysis Tools
            "TestComparisonTool": TestComparisonTool(),
            "TestCoverageAnalyzerTool": TestCoverageAnalyzerTool(),
            "GapAnalyzerTool": GapAnalyzerTool(),

            # Test Generation Tools
            "PytestGeneratorTool": PytestGeneratorTool(),
            "HTTPXTemplateTool": HTTPXTemplateTool(),
            "FixtureGeneratorTool": FixtureGeneratorTool(),
            #"TestModifierTool": lambda: TestModifierTool(),
            #"AssertionBuilderTool": lambda: AssertionBuilderTool(),
            
            # Xray Integration Tools
            "EnhancedXrayAPITool": EnhancedXrayAPITool(),

            # Jira Integration Tools
            #"JiraAPITool": lambda: JiraAPITool(),
            #"RequirementParserTool": lambda: RequirementParserTool(),
            
            # Validation Tools
            #"SyntaxValidatorTool": lambda: SyntaxValidatorTool(),
            #"DockerExecutorTool": lambda: DockerExecutorTool(),
            #"MockValidatorTool": lambda: MockValidatorTool(),
            #"RegressionCheckerTool": lambda: RegressionCheckerTool(),
            
            # Legacy Tools (placeholders for backward compatibility)
            #"ReadmeAnalyzerTool": lambda: ReadmeAnalyzerTool(),
            #"FilePatternAnalyzerTool": lambda: FilePatternAnalyzerTool(),
            #"TestDuplicationDetectorTool": lambda: TestDuplicationDetectorTool(),
            #"TestScenarioDesignerTool": lambda: TestScenarioDesignerTool(),
            #"TestOrganizationTool": lambda: TestOrganizationTool(),
            #"CoverageAnalyzerTool": lambda: CoverageAnalyzerTool()
        }
        """
    
    # Agent definitions
    @agent
    def bitbucket_analyst(self) -> Agent:
        config = self.agents_config['bitbucket_analyst']
        return Agent(
            config=config,
            tools=[
                self.EnhancedBitbucketCloneTool,
                self.BitbucketAnalyzerTool,
                #self.ReadmeAnalyzerTool,
                #self.FilePatternAnalyzerTool
            ],
            llm=self.llm
        )
    
    @agent
    def jira_requirements_analyst(self) -> Agent:
        config = self.agents_config['jira_requirements_analyst']
        return Agent(
            config=config,
            tools=[
                self.EnhancedXrayAPITool
            ],
            llm=self.llm
        )
    
    @agent
    def api_analyzer(self) -> Agent:
        config = self.agents_config['api_analyzer']
        return Agent(
            config=config,
            tools=[
                #self.FastAPIAnalyzerTool,
                self.OpenAPISpecReaderTool
            ],
            llm=self.llm
        )
    
    '''
    @agent
    def test_comparison_analyst(self) -> Agent:
        config = self.agents_config['test_comparison_analyst']
        return Agent(
            config=config,
            tools=[
                self.TestComparisonTool,
                self.GapAnalyzerTool,
                #self.TestDuplicationDetectorTool
            ],
            llm=self.llm
        )
    '''
    
    @agent
    def test_strategist(self) -> Agent:
        config = self.agents_config['test_strategist']
        return Agent(
            config=config,
            #tools=[
            #    self.TestScenarioDesignerTool,
            #    self.TestOrganizationTool,
            #    self.CoverageAnalyzerTool
            #],
            llm=self.llm
        )
    
    @agent
    def api_test_generator(self) -> Agent:
        config = self.agents_config['api_test_generator']
        return Agent(
            config=config,
            tools=[
                #self.PytestGeneratorTool,
                #self.HTTPXTemplateTool,
                #self.FixtureGeneratorTool,
                #self.LocalFileReaderTool, 
                self.TemplateFileReaderTool
            ],
            llm=self.llm
        )
    
    @agent
    def test_validator(self) -> Agent:
        config = self.agents_config['test_validator']
        return Agent(
            config=config,
            tools=[
                # Using available tools for validation
                self.TestComparisonTool,
                self.TestCoverageAnalyzerTool
            ],
            llm=self.llm
        )
    
    # Task definitions    
    @task
    def clone_and_analyze_repository(self) -> Task:
        return Task(
            config=self.tasks_config['clone_and_analyze_repository'],
            output_json=RepoAnalysisJson,
        )
    
    @task
    def extract_api_requirements(self) -> Task:
        return Task(
            config=self.tasks_config['extract_api_requirements'],
        )
    
    @task
    def analyze_api_endpoints(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_api_endpoints'],
            context=[self.extract_api_requirements()]
        )
    @task
    def design_api_test_scenarios(self) -> Task:
        return Task(
            config=self.tasks_config['design_api_test_scenarios'],
            context=[self.extract_api_requirements(), self.analyze_api_endpoints()],
            output_json=TestScenariosJson
        )
    
    #@task
    def generate_api_test_fixtures(self) -> Task:
        return Task(
            config=self.tasks_config['generate_api_test_fixtures'],
            context=[self.design_api_test_scenarios()]
        )
    
    @task
    def generate_or_modify_api_tests(self) -> Task:
        return Task(
            config=self.tasks_config['generate_or_modify_api_tests'],
            context=[self.extract_api_requirements(), 
                     self.analyze_api_endpoints(), 
                     self.design_api_test_scenarios()]
        )
    
    #@task
    def validate_modified_tests(self) -> Task:
        return Task(
            config=self.tasks_config['validate_modified_tests'],
            context=[self.generate_or_modify_api_tests()]
        )
    
    #@task
    def generate_integration_tests(self) -> Task:
        return Task(
            config=self.tasks_config['generate_integration_tests'],
            context=[self.validate_modified_tests()]
        )
    
    #@task
    def create_pull_request(self) -> Task:
        return Task(
            config=self.tasks_config['create_pull_request'],
            context=[self.generate_integration_tests()]
        )
    
    #@task
    def execute_and_report(self) -> Task:
        return Task(
            config=self.tasks_config['execute_and_report'],
            context=[self.create_pull_request()]
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the BackendAPITestGeneration crew"""
        return Crew(
            agents=self.agents,  # Automatically created by @agent decorator
            tasks=self.tasks,    # Automatically created by @task decorator
            process=Process.sequential,
            verbose=True
        )
