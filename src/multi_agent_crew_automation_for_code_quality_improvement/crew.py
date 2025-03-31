from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import GithubSearchTool
from crewai_tools import DirectorySearchTool
from crewai_tools import FileReadTool
from crewai_tools import CodeDocsSearchTool

@CrewBase
class MultiAgentCrewAutomationForCodeQualityImprovementCrew():
    """MultiAgentCrewAutomationForCodeQualityImprovement crew"""

    @agent
    def analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['analyzer'],
            tools=[GithubSearchTool(), DirectorySearchTool()],
        )

    @agent
    def refactor(self) -> Agent:
        return Agent(
            config=self.agents_config['refactor'],
            tools=[GithubSearchTool()],
        )

    @agent
    def test(self) -> Agent:
        return Agent(
            config=self.agents_config['test'],
            tools=[],
        )

    @agent
    def documentation(self) -> Agent:
        return Agent(
            config=self.agents_config['documentation'],
            tools=[FileReadTool(), CodeDocsSearchTool()],
        )

    @agent
    def security(self) -> Agent:
        return Agent(
            config=self.agents_config['security'],
            tools=[GithubSearchTool(), DirectorySearchTool()],
        )

    @agent
    def performance(self) -> Agent:
        return Agent(
            config=self.agents_config['performance'],
            tools=[GithubSearchTool(), DirectorySearchTool()],
        )


    @task
    def analyze_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_task'],
            tools=[GithubSearchTool(), DirectorySearchTool()],
        )

    @task
    def refactor_task(self) -> Task:
        return Task(
            config=self.tasks_config['refactor_task'],
            tools=[GithubSearchTool()],
        )

    @task
    def test_task(self) -> Task:
        return Task(
            config=self.tasks_config['test_task'],
            tools=[],
        )

    @task
    def documentation_task(self) -> Task:
        return Task(
            config=self.tasks_config['documentation_task'],
            tools=[FileReadTool(), CodeDocsSearchTool()],
        )

    @task
    def security_task(self) -> Task:
        return Task(
            config=self.tasks_config['security_task'],
            tools=[GithubSearchTool(), DirectorySearchTool()],
        )

    @task
    def performance_task(self) -> Task:
        return Task(
            config=self.tasks_config['performance_task'],
            tools=[GithubSearchTool(), DirectorySearchTool()],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the MultiAgentCrewAutomationForCodeQualityImprovement crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
