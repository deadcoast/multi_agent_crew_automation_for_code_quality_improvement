#!/usr/bin/env python
"""Main entry point for the MultiAgentCrewAutomationForCodeQualityImprovement project."""

import sys

from src.multi_agent_crew_automation_for_code_quality_improvement.crew import \
    MultiAgentCrewAutomationForCodeQualityImprovementCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def main():
    """Run the MultiAgentCrewAutomationForCodeQualityImprovement crew."""
    try:
        crew = MultiAgentCrewAutomationForCodeQualityImprovementCrew().crew()
        result = crew.kickoff()
        print("Crew execution completed.")
        print(result)
    except Exception as e:
        print(f"Error running crew: {e}")
        raise


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "repo_url": "sample_value",
        "analysis_threshold": "sample_value",
        "project_name": "sample_value",
        "test_command": "sample_value",
        "doc_path": "sample_value",
        "security_config": "sample_value",
        "performance_metrics": "sample_value",
    }
    try:
        MultiAgentCrewAutomationForCodeQualityImprovementCrew().crew().train(
            n_iterations=int(sys.argv[2]), filename=sys.argv[3], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}") from e


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MultiAgentCrewAutomationForCodeQualityImprovementCrew().crew().replay(
            task_id=sys.argv[2]
        )

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "repo_url": "sample_value",
        "analysis_threshold": "sample_value",
        "project_name": "sample_value",
        "test_command": "sample_value",
        "doc_path": "sample_value",
        "security_config": "sample_value",
        "performance_metrics": "sample_value",
    }
    try:
        MultiAgentCrewAutomationForCodeQualityImprovementCrew().crew().test(
            n_iterations=int(sys.argv[2]), inputs=inputs, eval_llm=sys.argv[3]
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}") from e


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        main()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
