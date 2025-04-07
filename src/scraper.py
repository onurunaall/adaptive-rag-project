"""
scraper.py

Implements a drop-shipping product discovery workflow using LangGraph agents.
The workflow scrapes provided web pages for detailed product information,
analyzes the scraped data, and generates insights on whether to start selling a product.
"""

import os
import operator
from typing import List, TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

# --- Import required components ---
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, END

# --- Define a custom AgentState type ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

# --- Global researcher tool instance ---
# This tool scrapes a list of URLs using WebBaseLoader and returns concatenated content.
@tool("Researcher")
def researcher_tool_impl(urls: List[str]) -> str:
    """
    Researcher tool: Given a list of URLs, scrape each web page and return the combined content.
    """
    try:
        # WebBaseLoader can accept a single URL or a list.
        loader = WebBaseLoader(urls)
        docs = loader.load()
        # Concatenate the content of all documents.
        combined = "\n\n".join(
            [f"<Document title='{doc.metadata.get('title', '')}'>\n{doc.page_content}\n</Document>" for doc in docs]
        )
        return combined
    except Exception as e:
        return f"Error in researcher_tool: {str(e)}"

# --- ScraperWorkflow Class ---
class ScraperWorkflow:
    def __init__(self,
                 model: str,
                 openai_api_key: str,
                 researcher_prompt: str = "You are an Amazon scraper. Extract detailed product information from the given URLs.",
                 analyzer_prompt: str = "You are a market analyst. Analyze the scraped product data and identify winning products.",
                 expert_prompt: str = "You are a drop-shipping expert. Based on the product data, advise whether to start selling the product.",
                 supervisor_prompt: str = "You are the supervisor over these agents: {agents}. Assign tasks based on conversation. Reply with an agent name for the next task, or 'FINISH' to end."
                 ):
        """
        Initialize the workflow with configuration parameters.
        """
        self.model = model
        self.openai_api_key = openai_api_key

        # Create a shared ChatOpenAI instance.
        self.llm = ChatOpenAI(model=self.model, openai_api_key=self.openai_api_key)

        # Store prompts as instance attributes.
        self.researcher_prompt = researcher_prompt
        self.analyzer_prompt = analyzer_prompt
        self.expert_prompt = expert_prompt
        self.supervisor_prompt = supervisor_prompt

        # Researcher tool is the only external tool; others rely solely on LLM reasoning.
        self.researcher_tool = researcher_tool_impl

        # Create agent executors only once.
        self.researcher_agent = self._create_agent([self.researcher_tool], self.researcher_prompt)
        self.analyzer_agent = self._create_agent([], self.analyzer_prompt)
        self.expert_agent = self._create_agent([], self.expert_prompt)

        # Create the supervisor runnable once.
        self.supervisor_runnable = self._create_supervisor(["RESEARCHER", "ANALYZER", "EXPERT"])

        # Precompile the workflow graph.
        self.compiled_graph = self.build_workflow().compile()

    def _create_agent(self, tools: List, system_prompt: str) -> AgentExecutor:
        """
        Helper method to create an agent executor.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools)

    def _create_supervisor(self, agents: List[str]):
        """
        Helper method to create the supervisor runnable.
        """
        options = ["FINISH"] + agents
        function_def = {
            "name": "supervisor",
            "description": "Select the next agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "next": {"anyOf": [{"enum": options}]},
                },
                "required": ["next"],
            },
        }
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.supervisor_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Choose one of: {options}."),
        ]).partial(options=str(options), agents=", ".join(agents))
        return prompt | self.llm.bind_functions(functions=[function_def], function_call="supervisor") | JsonOutputFunctionsParser()

    # --- Workflow Node Methods ---
    def researcher_node(self, state: AgentState) -> dict:
        """
        Invoke the researcher agent and return its output as a HumanMessage.
        """
        try:
            result = self.researcher_agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name="RESEARCHER")]}
        except Exception as e:
            return {"messages": [HumanMessage(content=f"Error in researcher_node: {str(e)}", name="RESEARCHER")]}

    def analyzer_node(self, state: AgentState) -> dict:
        """
        Invoke the analyzer agent to analyze scraped product data.
        """
        try:
            result = self.analyzer_agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name="ANALYZER")]}
        except Exception as e:
            return {"messages": [HumanMessage(content=f"Error in analyzer_node: {str(e)}", name="ANALYZER")]}

    def expert_node(self, state: AgentState) -> dict:
        """
        Invoke the expert agent to advise on product viability.
        """
        try:
            result = self.expert_agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name="EXPERT")]}
        except Exception as e:
            return {"messages": [HumanMessage(content=f"Error in expert_node: {str(e)}", name="EXPERT")]}

    def supervisor_node(self, state: AgentState) -> dict:
        """
        Invoke the supervisor to decide the next agent.
        """
        try:
            supervisor_result = self.supervisor_runnable.invoke(state)
            return {"next": supervisor_result["next"]}
        except Exception as e:
            return {"next": "FINISH", "error": f"Error in supervisor_node: {str(e)}"}

    def build_workflow(self) -> StateGraph:
        """
        Build and return the workflow graph.
        Flow:
          All agent nodes (RESEARCHER, ANALYZER, EXPERT) send output to SUPERVISOR,
          which then selects the next agent to run, or FINISH.
        """
        graph = StateGraph(AgentState)
        graph.add_node("RESEARCHER", self.researcher_node)
        graph.add_node("ANALYZER", self.analyzer_node)
        graph.add_node("EXPERT", self.expert_node)
        graph.add_node("SUPERVISOR", self.supervisor_node)
        graph.add_edge("RESEARCHER", "SUPERVISOR")
        graph.add_edge("ANALYZER", "SUPERVISOR")
        graph.add_edge("EXPERT", "SUPERVISOR")
        graph.add_conditional_edges(
            "SUPERVISOR",
            lambda state: state["next"],
            {
                "RESEARCHER": "RESEARCHER",
                "ANALYZER": "ANALYZER",
                "EXPERT": "EXPERT",
                "FINISH": END
            }
        )
        graph.set_entry_point("SUPERVISOR")
        return graph

    def run_workflow(self, data: AgentState) -> AgentState:
        """
        Run the workflow until completion using the precompiled graph.
        Returns the final state.
        """
        final_state = None
        for state in self.compiled_graph.stream(data):
            final_state = state
        return final_state

# --- Main Streamlit App ---
def main():
    import streamlit as st  # Import locally to decouple from core class.
    st.title("LangGraph + Function Call + Amazon Scraper")
    user_input = st.text_input("Enter your input here:")
    # For scraping, the user is expected to provide a comma-separated list of URLs.
    urls_input = st.text_input("Enter URLs (comma-separated):")
    process = st.button("Run Workflow")
    if process:
        if not urls_input:
            st.warning("Please provide at least one URL.")
            st.stop()
        # Parse URLs into a list.
        urls = [url.strip() for url in urls_input.split(",") if url.strip()]
        # Initialize the workflow with configuration from environment.
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("OPENAI_API_KEY not set in environment.")
            st.stop()
        workflow_instance = ScraperWorkflow(model="gpt-4", openai_api_key=openai_key)
        # Build initial state with the user's input and the list of URLs provided.
        # Here, we assume the researcher's role is to scrape the provided URLs.
        initial_state: AgentState = {"messages": [HumanMessage(content=", ".join(urls))], "next": ""}
        final_state = workflow_instance.run_workflow(initial_state)
        st.write("Final Output:")
        # Display the final output from the agent.
        if "agent_outcome" in final_state:
            st.write(final_state["agent_outcome"])
        else:
            st.write(final_state.get("next", "Workflow ended without output."))

if __name__ == "__main__":
    main()
