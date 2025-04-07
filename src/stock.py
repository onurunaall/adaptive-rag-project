"""
stock.py: Stock analysis agent workflow using LangGraph.

This script runs a multi-agent system to research stock news (via Yahoo Finance),
analyze the findings, and suggest potential trading strategies. It uses LangGraph
to manage the flow between agents and Streamlit for a simple UI.
"""

import os
import operator
from typing import List, TypedDict, Sequence
from dotenv import load_dotenv
import streamlit as st

# Pull in API keys from the .env file
load_dotenv()

# LangChain/LangGraph essentials
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable


# --- State ---
# Defines the data structure passed around the graph.
class AgentState(TypedDict):
    messages: Sequence[BaseMessage] # Conversation history + agent outputs
    next: str # Who runs next?


# --- Tools ---

# Keep a single instance of the Yahoo Finance tool to avoid re-initializing.
# Simple approach for this script; could use a class member in larger apps.
_yahoo_finance_tool = None

@tool("Trading_Research")
def get_stock_news(query: str) -> str:
    """Tool wrapper for fetching stock news from Yahoo Finance."""
    global _yahoo_finance_tool
    if _yahoo_finance_tool is None:
        _yahoo_finance_tool = YahooFinanceNewsTool()
    try:
        return _yahoo_finance_tool.run(query)
    except Exception as e:
        # Log the error and return a message usable by the agent graph.
        print(f"[Error] YahooFinanceNewsTool failed for query '{query}': {e}")
        return f"Error fetching news: {str(e)}"


# --- Workflow ---

class StockWorkflow:
    """Orchestrates the stock analysis agents using LangGraph."""

    # Agent identifiers used in the graph
    RESEARCHER = "RESEARCHER"
    ANALYZER = "ANALYZER"
    RECOMMENDER = "RECOMMENDER"
    SUPERVISOR = "SUPERVISOR"
    FINISH = "FINISH"

    def __init__(self, model: str, openai_api_key: str):
        """Sets up the LLM, agents, and supervisor for the workflow."""

        self.model = model
        self.openai_api_key = openai_api_key

        # One LLM instance for all agents
        self.llm = ChatOpenAI(model=self.model, openai_api_key=self.openai_api_key, temperature=0)

        self.agent_names = [self.RESEARCHER, self.ANALYZER, self.RECOMMENDER]

        # Define the available tools (only the researcher uses one here)
        self.tools = [get_stock_news]

        # --- Agent Definitions ---
        researcher_prompt = "You are a trader research assistant. Use the Trading_Research tool to find relevant, recent stock news based on the user's query."
        analyzer_prompt = "You are a market stock analyst. Review the research findings and conversation history. Identify key trends, risks, and potential opportunities based *only* on the provided information."
        recommender_prompt = "You are a trade recommender. Based *only* on the provided analysis and research, suggest potential trading strategies or points to consider. Frame these as possibilities, not direct financial advice."

        # Create the agent executors
        self.researcher_agent = self._create_agent(self.tools, researcher_prompt)
        self.analyzer_agent = self._create_agent([], analyzer_prompt) # No tools needed
        self.recommender_agent = self._create_agent([], recommender_prompt) # No tools needed

        # --- Supervisor Definition ---
        supervisor_prompt_template = (
            "You are the supervisor managing these agents: {agents}. "
            "Based on the conversation, decide who should act next. "
            "Respond with only the agent's name or '{finish_step}' if the task is complete."
        )
        formatted_supervisor_prompt = supervisor_prompt_template.format(
            agents=", ".join(self.agent_names), finish_step=self.FINISH
        )
        self.supervisor_runnable = self._create_supervisor(self.agent_names, formatted_supervisor_prompt)

    def _create_agent(self, tools: List, system_prompt: str) -> AgentExecutor:
        """Builds an agent executor instance."""
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"), # History/context
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Tool results/thoughts
        ])
        agent = create_openai_tools_agent(self.llm, tools, agent_prompt)
        # Set verbose=True to see agent thoughts/tool calls during execution
        return AgentExecutor(agent=agent, tools=tools, verbose=False)

    def _create_supervisor(self, agents: List[str], system_prompt: str) -> Runnable:
        """Creates the supervisor runnable using OpenAI functions for routing."""

        # Options the supervisor can choose from
        supervisor_options = [self.FINISH] + agents

        # Define the function structure the LLM must use to respond.
        # This forces the LLM to choose one of the valid next steps.
        route_action_function = {
            "name": "route_action",
            "description": "Select the next agent or finish.",
            "parameters": {
                "type": "object",
                "properties": {"next": {"anyOf": [{"enum": supervisor_options}]}},
                "required": ["next"],
            },
        }

        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", f"Select the next step. Your options are: {supervisor_options}"),
        ])

        # Chain: Prompt -> LLM (forced to call 'route_action') -> Parse the function output
        return (
            supervisor_prompt
            | self.llm.bind_functions(functions=[route_action_function], function_call="route_action")
            | JsonOutputFunctionsParser()
        )

    # --- Graph Nodes ---
    # Each node corresponds to an agent or the supervisor running.

    def _execute_agent_node(self, state: AgentState, agent: AgentExecutor, agent_name: str) -> dict:
        """Helper to run an agent and format the output for the graph state."""
        try:
            result = agent.invoke(state)
            # Wrap the output in a HumanMessage to add it to the conversation history
            return {"messages": [HumanMessage(content=result["output"], name=agent_name)]}
        except Exception as e:
            error_msg = f"Error in {agent_name}: {str(e)}"
            print(error_msg)
            # Still return a message so the graph can potentially continue/report the error
            return {"messages": [HumanMessage(content=error_msg, name=agent_name)]}

    def run_researcher(self, state: AgentState) -> dict:
        """Node that executes the researcher agent."""
        print(f"---> Executing {self.RESEARCHER}")
        return self._execute_agent_node(state, self.researcher_agent, self.RESEARCHER)

    def run_analyzer(self, state: AgentState) -> dict:
        """Node that executes the analyzer agent."""
        print(f"---> Executing {self.ANALYZER}")
        return self._execute_agent_node(state, self.analyzer_agent, self.ANALYZER)

    def run_recommender(self, state: AgentState) -> dict:
        """Node that executes the recommender agent."""
        print(f"---> Executing {self.RECOMMENDER}")
        return self._execute_agent_node(state, self.recommender_agent, self.RECOMMENDER)

    def run_supervisor(self, state: AgentState) -> dict:
        """Node that executes the supervisor to decide the next step."""
        print(f"---> Executing {self.SUPERVISOR}")
        try:
            supervisor_decision = self.supervisor_runnable.invoke(state)
            next_step = supervisor_decision.get("next", self.FINISH) # Default to FINISH if key missing
            print(f"Supervisor Decision: Next step = {next_step}")
            return {"next": next_step}
        except Exception as e:
            # If supervisor fails, we probably should just end.
            error_msg = f"Error in Supervisor: {str(e)}. Ending workflow."
            print(error_msg)
            return {"next": self.FINISH}

    def build_graph(self) -> StateGraph:
        """Constructs the LangGraph StateGraph."""
        graph = StateGraph(AgentState)

        # Add nodes for each agent and the supervisor
        graph.add_node(self.RESEARCHER, self.run_researcher)
        graph.add_node(self.ANALYZER, self.run_analyzer)
        graph.add_node(self.RECOMMENDER, self.run_recommender)
        graph.add_node(self.SUPERVISOR, self.run_supervisor)

        # Edges: All agents report back to the supervisor
        graph.add_edge(self.RESEARCHER, self.SUPERVISOR)
        graph.add_edge(self.ANALYZER, self.SUPERVISOR)
        graph.add_edge(self.RECOMMENDER, self.SUPERVISOR)

        # Conditional edges: The supervisor routes to the next agent or ends
        graph.add_conditional_edges(
            self.SUPERVISOR,
            lambda state: state["next"], # Route based on the 'next' value in the state
            {
                self.RESEARCHER: self.RESEARCHER,
                self.ANALYZER: self.ANALYZER,
                self.RECOMMENDER: self.RECOMMENDER,
                self.FINISH: END # Special END state from LangGraph
            }
        )

        # The supervisor is the entry point
        graph.set_entry_point(self.SUPERVISOR)

        # Compile the graph into a runnable application
        compiled_graph = graph.compile()
        return compiled_graph


# --- Streamlit UI ---

def run_ui():
    """Runs the Streamlit interface for the stock workflow."""
    st.set_page_config(page_title="Stock Analysis Workflow", layout="wide")
    st.title("Stock Analysis Workflow 📈")
    st.caption("Agents researching and analyzing stock news")

    # --- Sidebar Config ---
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox(
            "OpenAI Model",
            ["gpt-4-turbo", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
            index=0
        )
        api_key_input = st.text_input("OpenAI API Key", type="password")

    # --- Main Input/Output ---
    query = st.text_input("What stocks are you interested in?", placeholder="e.g., Latest news for MSFT and GOOGL")

    if st.button("Run Workflow"):
        if not api_key_input:
            st.warning("Please enter your OpenAI API key in the sidebar.", icon="🔑")
            st.stop()
        if not query:
            st.warning("Please enter a stock query.", icon="❓")
            st.stop()

        # Initialize and run the workflow
        with st.spinner("🤖 Agents at work..."):
            try:
                workflow = StockWorkflow(model=model_choice, openai_api_key=api_key_input)
                graph_app = workflow.build_graph()

                initial_state = {"messages": [HumanMessage(content=query)], "next": ""}

                st.subheader("Workflow Progress:")
                log_container = st.container(height=400) # Container for scrolling logs

                final_state = None
                for step_output in graph_app.stream(initial_state):
                    node_name = list(step_output.keys())[0]
                    node_data = step_output[node_name]

                    log_container.markdown(f"**Running Node:** `{node_name}`")
                    log_container.write(node_data) # Show state changes/messages
                    log_container.divider()
                    final_state = node_data # Keep track of last state data

                st.success("Workflow complete!", icon="✅")

                # Display the final message from the conversation history
                # if final_state and "messages" in final_state and final_state["messages"]:
                #     st.subheader("Final Output:")
                #     st.info(final_state["messages"][-1].content)

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")


if __name__ == "__main__":
    run_ui()
