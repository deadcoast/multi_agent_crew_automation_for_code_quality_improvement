"""
This file contains the crew for the MultiAgentCrewAutomationForCodeQualityImprovement project.
"""

import os

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    CodeDocsSearchTool,
    DirectorySearchTool,
    FileReadTool,
    GithubSearchTool,
)
from dotenv import load_dotenv

# Import our new advanced math tools
from src.math_tools2 import (
    AlgorithmQualityAssessmentTool,
    APIAnalysisTool,
    CodeMathRepresentationTool,
    CodeVectorizationTool,
)

# Import our algorithm tools
from src.multi_agent_crew_automation_for_code_quality_improvement.algorithm_tools import (
    AlgorithmIntegrationTool,
    BayesianOptimizationTool,
    ConditionalRandomFieldTool,
    ExtremeLearningMachineTool,
    HopfieldNetworkTool,
    IsolationForestTool,
    SelfOrganizingMapTool,
    SymbolicRegressionTool,
    TsetlinMachineTool,
)

# Import our custom math tools
from src.multi_agent_crew_automation_for_code_quality_improvement.math_tools import (
    CodeClusteringTool,
    CodeComplexityAnalysisTool,
    DependencyGraphAnalysisTool,
    DocumentationAnalysisTool,
    PerformanceAnalysisTool,
    SecurityAnalysisTool,
)

# Load environment variables
load_dotenv()

@CrewBase
class MultiAgentCrewAutomationForCodeQualityImprovementCrew():
    """MultiAgentCrewAutomationForCodeQualityImprovement crew"""

    def __init__(self):
        """Initialize the crew with agent and task configurations."""
        self.agents_config = self._load_config('agents.yaml')
        self.tasks_config = self._load_config('tasks.yaml')

    def _load_config(self, filename):
        """Load configuration from YAML file."""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'config',
            filename
        )
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @agent
    def analyzer(self) -> Agent:
        return Agent(
            role=self.agents_config['analyzer']['role'],
            goal=self.agents_config['analyzer']['goal'],
            backstory=self.agents_config['analyzer']['backstory'],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["file", "issue", "pull"]
                ),
                DirectorySearchTool(),
                # Add mathematical analysis tools
                CodeComplexityAnalysisTool(),
                DependencyGraphAnalysisTool(),
                CodeClusteringTool(),
                # Add specialized algorithms
                ExtremeLearningMachineTool(),  # For pattern recognition in code
                IsolationForestTool(),         # For anomaly detection
                TsetlinMachineTool(),          # For interpretable pattern learning
                AlgorithmIntegrationTool()     # Meta-tool for algorithm selection
            ],
        )

    @agent
    def refactor(self) -> Agent:
        return Agent(
            role=self.agents_config['refactor']['role'],
            goal=self.agents_config['refactor']['goal'],
            backstory=self.agents_config['refactor']['backstory'],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["file", "issue", "pull"]
                ),
                # Add mathematical refactoring tools
                CodeComplexityAnalysisTool(),
                CodeClusteringTool(),
                # Add new advanced math tools
                CodeMathRepresentationTool(),        # Translate code to mathematical representation
                CodeVectorizationTool(),             # Convert to vector embeddings
                AlgorithmQualityAssessmentTool(),    # Assess algorithm quality in 3D space
                APIAnalysisTool(),                   # Analyze API quality mathematically
                # Add specialized algorithms
                SymbolicRegressionTool(),     # For discovering relationships in code metrics
                BayesianOptimizationTool(),   # For optimizing refactoring decisions
                AlgorithmIntegrationTool()    # Meta-tool for algorithm selection
            ],
        )

    @agent
    def test(self) -> Agent:
        return Agent(
            role=self.agents_config['test']['role'],
            goal=self.agents_config['test']['goal'],
            backstory=self.agents_config['test']['backstory'],
            tools=[
                # Add dependency graph analysis for test selection
                DependencyGraphAnalysisTool(),
                # Add specialized algorithms
                ConditionalRandomFieldTool(),  # For test dependency analysis
                AlgorithmIntegrationTool()     # Meta-tool for algorithm selection
            ],
        )

    @agent
    def documentation(self) -> Agent:
        return Agent(
            role=self.agents_config['documentation']['role'],
            goal=self.agents_config['documentation']['goal'],
            backstory=self.agents_config['documentation']['backstory'],
            tools=[
                FileReadTool(),
                CodeDocsSearchTool(),
                # Add documentation analysis with NLP
                DocumentationAnalysisTool(),
                # Add specialized algorithms
                HopfieldNetworkTool(),        # For pattern association in documentation
                SelfOrganizingMapTool(),      # For document clustering
                AlgorithmIntegrationTool()    # Meta-tool for algorithm selection
            ],
        )

    @agent
    def security(self) -> Agent:
        return Agent(
            role=self.agents_config['security']['role'],
            goal=self.agents_config['security']['goal'],
            backstory=self.agents_config['security']['backstory'],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["file", "issue", "pull"]
                ),
                DirectorySearchTool(),
                # Add security analysis tool
                SecurityAnalysisTool(),
                # Add specialized algorithms
                IsolationForestTool(),         # For detecting security anomalies
                TsetlinMachineTool(),          # For learning security patterns
                AlgorithmIntegrationTool()     # Meta-tool for algorithm selection
            ],
        )

    @agent
    def performance(self) -> Agent:
        return Agent(
            role=self.agents_config['performance']['role'],
            goal=self.agents_config['performance']['goal'],
            backstory=self.agents_config['performance']['backstory'],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["file", "issue", "pull"]
                ),
                DirectorySearchTool(),
                # Add performance analysis tool
                PerformanceAnalysisTool(),
                # Add specialized algorithms
                BayesianOptimizationTool(),    # For performance optimization
                SymbolicRegressionTool(),      # For understanding performance relationships
                AlgorithmIntegrationTool()     # Meta-tool for algorithm selection
            ],
        )

    @task
    def analyze_task(self) -> Task:
        return Task(
            description=self.tasks_config['analyze_task']['description'],
            expected_output=self.tasks_config['analyze_task']['expected_output'],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["file", "issue", "pull"]
                ),
                DirectorySearchTool(),
                # Add mathematical analysis tools
                CodeComplexityAnalysisTool(),
                DependencyGraphAnalysisTool(),
                CodeClusteringTool(),
                # Add specialized algorithms
                ExtremeLearningMachineTool(),  # For pattern recognition in code
                IsolationForestTool(),         # For anomaly detection
                TsetlinMachineTool(),          # For interpretable pattern learning
                AlgorithmIntegrationTool()     # Meta-tool for algorithm selection
            ],
            agent=self.analyzer()
        )

    @task
    def refactor_task(self) -> Task:
        return Task(
            description=self.tasks_config['refactor_task']['description'],
            expected_output=self.tasks_config['refactor_task']['expected_output'],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["file", "issue", "pull"]
                ),
                # Add mathematical refactoring tools
                CodeComplexityAnalysisTool(),
                CodeClusteringTool(),
                # Add new advanced math tools
                CodeMathRepresentationTool(),        # Translate code to mathematical representation
                CodeVectorizationTool(),             # Convert to vector embeddings
                AlgorithmQualityAssessmentTool(),    # Assess algorithm quality in 3D space
                APIAnalysisTool(),                   # Analyze API quality mathematically
                # Add specialized algorithms
                SymbolicRegressionTool(),     # For discovering relationships in code metrics
                BayesianOptimizationTool(),   # For optimizing refactoring decisions
                AlgorithmIntegrationTool()    # Meta-tool for algorithm selection
            ],
            agent=self.refactor(),
            context=[self.analyze_task()]
        )

    @task
    def test_task(self) -> Task:
        return Task(
            description=self.tasks_config['test_task']['description'],
            expected_output=self.tasks_config['test_task']['expected_output'],
            tools=[
                # Add dependency graph analysis for test selection
                DependencyGraphAnalysisTool(),
                # Add specialized algorithms
                ConditionalRandomFieldTool(),  # For test dependency analysis
                AlgorithmIntegrationTool()     # Meta-tool for algorithm selection
            ],
            agent=self.test(),
            context=[self.refactor_task()]
        )

    @task
    def documentation_task(self) -> Task:
        return Task(
            description=self.tasks_config['documentation_task']['description'],
            expected_output=self.tasks_config['documentation_task']['expected_output'],
            tools=[
                FileReadTool(),
                CodeDocsSearchTool(),
                # Add documentation analysis with NLP
                DocumentationAnalysisTool(),
                # Add specialized algorithms
                HopfieldNetworkTool(),        # For pattern association in documentation
                SelfOrganizingMapTool(),      # For document clustering
                AlgorithmIntegrationTool()    # Meta-tool for algorithm selection
            ],
            agent=self.documentation(),
            context=[self.refactor_task()]
        )

    @task
    def security_task(self) -> Task:
        return Task(
            description=self.tasks_config['security_task']['description'],
            expected_output=self.tasks_config['security_task']['expected_output'],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["file", "issue", "pull"]
                ),
                DirectorySearchTool(),
                # Add security analysis tool
                SecurityAnalysisTool(),
                # Add specialized algorithms
                IsolationForestTool(),         # For detecting security anomalies
                TsetlinMachineTool(),          # For learning security patterns
                AlgorithmIntegrationTool()     # Meta-tool for algorithm selection
            ],
            agent=self.security(),
            context=[self.analyze_task(), self.refactor_task()]
        )

    @task
    def performance_task(self) -> Task:
        return Task(
            description=self.tasks_config['performance_task']['description'],
            expected_output=self.tasks_config['performance_task']['expected_output'],
            tools=[
                GithubSearchTool(
                    gh_token=os.getenv("GITHUB_TOKEN"),
                    content_types=["file", "issue", "pull"]
                ),
                DirectorySearchTool(),
                # Add performance analysis tool
                PerformanceAnalysisTool(),
                # Add specialized algorithms
                BayesianOptimizationTool(),    # For performance optimization
                SymbolicRegressionTool(),      # For understanding performance relationships
                AlgorithmIntegrationTool()     # Meta-tool for algorithm selection
            ],
            agent=self.performance(),
            context=[self.refactor_task(), self.test_task()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MultiAgentCrewAutomationForCodeQualityImprovement crew"""
        return Crew(
            agents=[
                self.analyzer(),
                self.refactor(),
                self.test(),
                self.documentation(),
                self.security(),
                self.performance()
            ],
            tasks=[
                self.analyze_task(),
                self.refactor_task(),
                self.test_task(),
                self.documentation_task(),
                self.security_task(),
                self.performance_task()
            ],
            process=Process.sequential,
            verbose=True,
        )
