import os
from datetime import datetime, timedelta

import requests

# Use CrewAI and import from crewai.tools
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool  # Use CrewAI's own tool system
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from opensearch_client import get_opensearch_client


# Check if Ollama is available and get available models
def check_ollama_availability():
    """Check if Ollama is running and return available models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model.get("name") for model in models if model.get("name")]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []


# Test model with a simple query to verify it works
def test_model(model_name):
    """Test if the model can respond to a simple prompt."""
    try:
        llm = OllamaLLM(model=model_name, temperature=0.2)
        prompt = ChatPromptTemplate.from_template("Say hello!")
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({})
        return bool(result)
    except Exception as e:
        print(f"Error testing model {model_name}: {e}")
        return False


# Define custom tools by extending BaseTool from CrewAI
class SearchPatentsTool(BaseTool):
    name: str = "search_patents"
    description: str = "Search for patents matching a query"

    def _run(self, query: str, top_k: int = 20) -> str:
        client = get_opensearch_client("localhost", 9200)
        index_name = "patents"

        search_query = {
            "size": top_k,
            "query": {"bool": {"must": [{"match": {"abstract": query}}]}},
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        try:
            response = client.search(index=index_name, body=search_query)
            results = response["hits"]["hits"]

            # Format results as a string for better LLM consumption
            formatted_results = []
            for i, hit in enumerate(results):
                source = hit["_source"]
                formatted_results.append(
                    f"{i+1}. Title: {source.get('title', 'N/A')}\n"
                    f"   Date: {source.get('publication_date', 'N/A')}\n"
                    f"   Patent ID: {source.get('patent_id', 'N/A')}\n"
                    f"   Abstract: {source.get('abstract', 'N/A')[:200]}...\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error searching patents: {str(e)}"


class SearchPatentsByDateRangeTool(BaseTool):
    name: str = "search_patents_by_date_range"
    description: str = "Search for patents in a specific date range"

    def _run(self, query: str, start_date: str, end_date: str, top_k: int = 30) -> str:
        client = get_opensearch_client("localhost", 9200)
        index_name = "patents"

        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"abstract": query}}],
                    "filter": [
                        {
                            "range": {
                                "publication_date": {"gte": start_date, "lte": end_date}
                            }
                        }
                    ],
                }
            },
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        try:
            response = client.search(index=index_name, body=search_query)
            results = response["hits"]["hits"]

            # Format results as a string
            formatted_results = []
            for i, hit in enumerate(results):
                source = hit["_source"]
                formatted_results.append(
                    f"{i+1}. Title: {source.get('title', 'N/A')}\n"
                    f"   Date: {source.get('publication_date', 'N/A')}\n"
                    f"   Patent ID: {source.get('patent_id', 'N/A')}\n"
                    f"   Abstract: {source.get('abstract', 'N/A')[:200]}...\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error searching patents: {str(e)}"


class AnalyzePatentTrendsTool(BaseTool):
    name: str = "analyze_patent_trends"
    description: str = "Analyze trends in patent data"

    def _run(self, patents_data: str) -> str:
        # In a real implementation, this would use NLP to analyze trends
        # Here, we just return the input data for simplicity
        return f"Analysis of patent trends: {patents_data}"


# Define our agents
def create_patent_analysis_crew(model_name="llama3"):
    """
    Create a CrewAI crew for patent analysis using Ollama.

    Args:
        model_name: Name of the Ollama model to use

    Returns:
        Crew: A CrewAI crew configured for patent analysis
    """
    # Check if model exists in Ollama
    available_models = check_ollama_availability()
    if not available_models:
        raise RuntimeError(
            "Ollama service is not available. Make sure Ollama is running."
        )

    # Test model
    if not test_model(model_name):
        raise RuntimeError(f"Model {model_name} is not responding to test prompts.")
    
    print("model found and tested successfully")

    # Fix the model format by adding the 'ollama/' prefix
    if not model_name.startswith("ollama/"):
        model_name = f"ollama/{model_name}"

    llm = OllamaLLM(model=model_name, temperature=0.2)

    # Create tools using CrewAI's BaseTool subclasses
    tools = [
        SearchPatentsTool(),
        SearchPatentsByDateRangeTool(),
        AnalyzePatentTrendsTool(),
    ]

    # Create agents with the correct tools
    research_director = Agent(
        role="Research Director",
        goal="Coordinate research efforts and define the scope of patent analysis",
        backstory="You are an experienced research director who specializes in technological innovation analysis.",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools=tools,
    )

    patent_retriever = Agent(
        role="Patent Retriever",
        goal="Find and retrieve the most relevant patents related to the research area",
        backstory="You are a specialized patent researcher with expertise in information retrieval systems.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    data_analyst = Agent(
        role="Patent Data Analyst",
        goal="Analyze patent data to identify trends, patterns, and emerging technologies",
        backstory="You are a data scientist specializing in patent analysis with years of experience in technology forecasting.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    innovation_forecaster = Agent(
        role="Innovation Forecaster",
        goal="Predict future innovations and technologies based on patent trends",
        backstory="You are an expert in technological forecasting with a track record of accurate predictions in emerging technologies.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    # Create tasks with shorter, simpler descriptions (to reduce LLM load)
    task1 = Task(
        description="""
        Define a research plan for lithium battery patents:
        1. Key technology areas to focus on
        2. Time periods for analysis (focus on last 3 years)
        3. Specific technological aspects to analyze
        """,
        expected_output="""A research plan with focus areas, time periods, and key technological aspects.""",
        agent=research_director,
    )

    task2 = Task(
        description="""
        Using the research plan, retrieve patents related to lithium battery technology from the last 3 years.
        Use the search_patents and search_patents_by_date_range tools to gather comprehensive data.
        Focus on the most relevant and innovative patents.
        Group patents by sub-technologies within lithium batteries.
        Provide a summary of the retrieved patents, including:
        - Total number of patents found
        - Key companies/assignees
        - Main technological categories
        """,
        expected_output="""A comprehensive patent retrieval report containing:
        - Summary of total patents found
        - List of key patents grouped by sub-technology
        - Analysis of top companies/assignees
        - Overview of main technological categories
        - List of the most innovative patents with summaries
        """,
        agent=patent_retriever,
        dependencies=[task1],
    )

    task3 = Task(
        description="""
        Analyze the retrieved patent data to identify trends and patterns:
        1. Identify growing vs. declining areas of innovation
        2. Analyze technology evolution over time
        3. Identify key companies and their focus areas
        4. Determine emerging sub-technologies within lithium batteries
        5. Analyze patent claims to understand technological improvements
        
        Create a comprehensive analysis with specific trends, supported by data.
        """,
        expected_output="""A trend analysis report containing:
        - Identification of growing vs. declining technology areas
        - Timeline of technology evolution
        - Company focus analysis
        - Emerging sub-technologies list
        - Technical improvement trends
        - Data-backed conclusions on innovation patterns
        """,
        agent=data_analyst,
        dependencies=[task2],
    )

    task4 = Task(
        description="""
        Based on the patent analysis, predict future innovations in lithium battery technology:
        1. Identify technologies likely to see breakthroughs in the next 2-3 years
        2. Recommend specific areas for R&D investment
        3. Predict which companies are positioned to lead innovation
        4. Identify potential disruptive technologies
        5. Outline specific technical improvements likely to emerge
        
        Create a detailed forecast with specific technology predictions and justification.
        """,
        expected_output="""A future innovation forecast containing:
        - Predicted breakthrough technologies for next 2-3 years
        - Prioritized list of R&D investment areas
        - Companies likely to lead future innovation
        - Potential disruptive technologies and their impact
        - Timeline of expected technical improvements
        - Justification for all predictions based on patent data
        """,
        agent=innovation_forecaster,
        dependencies=[task3],
    )

    # Create the crew with debugging enabled
    crew = Crew(
        agents=[
            research_director,
            patent_retriever,
            data_analyst,
            innovation_forecaster,
        ],
        tasks=[task1, task2, task3, task4],
        verbose=True,
        process=Process.sequential,
        cache=False,  # Disable cache to prevent issues
    )

    return crew


def run_patent_analysis(research_area="Lithium Battery", model_name="llama3"):
    """
    Run the patent analysis crew for the specified research area.

    Args:
        research_area (str): The research area to analyze (e.g., "Lithium Battery")
        model_name (str): Ollama model to use

    Returns:
        str: Analysis results
    """
    try:
        crew = create_patent_analysis_crew(model_name)
        result = crew.kickoff(inputs={"research_area": research_area})

        # Extract the string output from the CrewOutput object
        if hasattr(result, "output"):
            # Recent CrewAI versions store results in the 'output' attribute
            return result.output
        elif hasattr(result, "result"):
            # Some versions might use 'result'
            return result.result
        else:
            # Last resort - convert to string
            return str(result)
    except Exception as e:
        return (
            f"Analysis failed: {str(e)}\n\nTroubleshooting tips:\n"
            + "1. Make sure Ollama is running: 'ollama serve'\n"
            + "2. Pull a compatible model: 'ollama pull llama3' or 'ollama pull mistral'\n"
            + "3. Check Ollama logs for errors\n"
            + "4. Try a simpler model or reduce task complexity"
        )


if __name__ == "__main__":
    # Get the research area from user input
    research_area = input(
        "Enter the research area to analyze (default: Lithium Battery): "
    )
    if not research_area:
        research_area = "Lithium Battery"

    # Get the model name from user input
    model_name = input("Enter the Ollama model to use (default: llama2): ")
    if not model_name:
        model_name = "llama2"

    # Run the analysis
    result = run_patent_analysis(research_area, model_name)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patent_analysis_{timestamp}.txt"

    # Ensure result is a string before writing to file
    if not isinstance(result, str):
        result = str(result)

    with open(filename, "w") as f:
        f.write(result)

    print(f"Analysis completed and saved to {filename}")
