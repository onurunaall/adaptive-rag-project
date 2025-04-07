"""
stock.py

Implements a stock market analysis workflow using LangGraph agents.
It leverages YahooFinanceNewsTool to research stock news, and uses ChatOpenAI to analyze market data
and provide trade recommendations (as suggestions, not actual trades). This workflow is integrated
into a Streamlit app.
"""

import os
import operator
from typing import Sequence, TypedDict, List
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file.
load_dotenv()

# --- Import required LangChain and LangGraph components ---
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable

# --- Define a custom AgentState type for the workflow ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

# --- Global Instance for YahooFinanceNewsTool ---
_yahoo_tool_instance = None

@tool("Trading_Research")
def researcher_tool_impl(query: str) -> str:
    """
    Researcher tool: Uses YahooFinanceNewsTool to fetch stock news.
    This function instantiates YahooFinanceNewsTool only once.
    """
    global _yahoo_tool_instance
    if _yahoo_tool_instance is None:
        _yahoo_tool_instance = YahooFinanceNewsTool()
    try:
        return _yahoo_tool_instance.run(query)
    except Exception as e:
        return f"Error in researcher_tool: {str(e)}"

# --- StockWorkflow Class ---
class StockWorkflow:
    def __init__(
        self,
        model: str,
        openai_api_key: str,
        researcher_prompt: str = "You are a trader research assistant. Use Yahoo Finance News to gather accurate stock news.",
        analyzer_prompt: str = "You are a market stock analyst. Analyze the market data and advise on investment opportunities.",
        recommender_prompt: str = "You are a trade recommender. Based on market analysis, suggest optimal trading strategies.",
        supervisor_prompt: str = "You are the supervisor over the following agents: {agents}. Assign tasks based on the conversation. Reply with the agent name for the next task, or 'FINISH' if complete."
    ):
        """
        Initialize the workflow with configuration parameters.
        """
        self.model = model
        self.openai_api_key = openai_api_key
        
        # Create one shared LLM instance for all agents.
        self.llm = ChatOpenAI(model=self.model, openai_api_key=self.openai_api_key)
        
        # Store prompts as instance attributes.
        self.researcher_prompt = researcher_prompt
        self.analyzer_prompt = analyzer_prompt
        self.recommender_prompt = recommender_prompt
        self.supervisor_prompt = supervisor_prompt
        
        # Set up tool: Only the researcher tool is needed.
        self.researcher_tool = researcher_tool_impl

        # Create agent executors only once.
        self.researcher_agent = self._create_agent([self.researcher_tool], self.researcher_prompt)
        # For analyzer and recommender agents, no external tools are needed.
        self.analyzer_agent = self._create_agent([], self.analyzer_prompt)
        self.recommender_agent = self._create_agent([], self.recommender_prompt)
        
        # Create the supervisor runnable once.
        self.supervisor_runnable = self._create_supervisor(["RESEARCHER", "ANALYZER", "RECOMMENDER"])

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
        Invoke the analyzer agent and return its output as a HumanMessage.
        This agent uses the LLM to analyze the provided data.
        """
        try:
            result = self.analyzer_agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name="ANALYZER")]}
        except Exception as e:
            return {"messages": [HumanMessage(content=f"Error in analyzer_node: {str(e)}", name="ANALYZER")]}

    def recommender_node(self, state: AgentState) -> dict:
        """
        Invoke the recommender agent and return its output as a HumanMessage.
        This agent uses the LLM to generate trade recommendations.
        """
        try:
            result = self.recommender_agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name="RECOMMENDER")]}
        except Exception as e:
            return {"messages": [HumanMessage(content=f"Error in recommender_node: {str(e)}", name="RECOMMENDER")]}

    def supervisor_node(self, state: AgentState) -> dict:
        """
        Invoke the supervisor to decide the next agent.
        Returns a dict with a key "next" indicating the agent to run next.
        """
        try:
            supervisor_result = self.supervisor_runnable.invoke(state)
            return {"next": supervisor_result["next"]}
        except Exception as e:
            return {"next": "FINISH", "error": f"Error in supervisor_node: {str(e)}"}

    def build_workflow(self) -> StateGraph:
        """
        Build and compile the workflow graph.
        All agent nodes send their output to the supervisor, which then chooses the next agent.
        """
        workflow = StateGraph(AgentState)
        # Add nodes.
        workflow.add_node("RESEARCHER", self.researcher_node)
        workflow.add_node("ANALYZER", self.analyzer_node)
        workflow.add_node("RECOMMENDER", self.recommender_node)
        workflow.add_node("SUPERVISOR", self.supervisor_node)
        # All agent nodes lead to the supervisor.
        workflow.add_edge("RESEARCHER", "SUPERVISOR")
        workflow.add_edge("ANALYZER", "SUPERVISOR")
        workflow.add_edge("RECOMMENDER", "SUPERVISOR")
        # Conditional edge based on supervisor output.
        workflow.add_conditional_edges(
            "SUPERVISOR",
            lambda state: state["next"],
            {
                "RESEARCHER": "RESEARCHER",
                "ANALYZER": "ANALYZER",
                "RECOMMENDER": "RECOMMENDER",
                "FINISH": END
            }
        )
        workflow.set_entry_point("SUPERVISOR")
        return workflow

# --- Main Streamlit App ---
def main():
    st.title("LangGraph + Function Call + YahooFinance 👾")
    # Gather configuration from UI.
    model = st.sidebar.selectbox(
        "Select Model",
        ["gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]
    )
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if not api_key:
        st.error("Please provide an API key.")
        st.stop()
    user_input = st.text_input("Enter your input here:")
    if st.button("Run Workflow"):
        with st.spinner("Running Workflow..."):
            # Instantiate the workflow with configuration from UI.
            workflow_instance = StockWorkflow(model=model, openai_api_key=api_key)
            workflow = workflow_instance.build_workflow()
            # Initial state: start with user's message.
            initial_state: AgentState = {"messages": [HumanMessage(content=user_input)], "next": ""}
            for state in workflow.stream(initial_state):
                if "__end__" not in state:
                    st.write(state)
                    st.write("-----")

if __name__ == "__main__":
    main()
