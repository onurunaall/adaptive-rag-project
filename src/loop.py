"""
loop.py

Implements an agent loop workflow using LangGraph.
The workflow repeatedly invokes a JSON-based chat agent (created via create_json_chat_agent)
and then executes tool actions as directed until the agent finishes.
"""

import json
import os
import operator
from typing import List, Union, TypedDict

from langchain.agents import create_json_chat_agent, AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage, ToolMessage, BaseMessage
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph

class AgentLoopState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: List[tuple[AgentAction, str]]

class AgentLoopWorkflow:
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.openai_api_key = openai_api_key
        self.model = model

        self.search_tool = TavilySearchResults(max_results=1)
        self.repl = PythonREPL()

        self.chat_model = ChatOpenAI(openai_api_key=self.openai_api_key, model=self.model)
        self.python_repl_tool = self._build_python_repl_tool()

        self.tools = [self.search_tool, self.python_repl_tool]

        self.prompt = ChatPromptTemplate(input_variables=["agent_scratchpad", "input", "tool_names", "tools"],
                                         messages=[SystemMessage(content="Assistant is a large language model trained by OpenAI. It can answer questions and use tools."),
                                                   MessagesPlaceholder(variable_name="chat_history", optional=True),
                                                   HumanMessage(content=("TOOLS\n------\n"
                                                                         "Available tools: {tools}\n\n"
                                                                         "When responding, output a JSON with an action and action_input.\n\n"
                                                                         "USER'S INPUT\n--------------------\n{input}")),
                                                   MessagesPlaceholder(variable_name="agent_scratchpad")])

        self.agent_runnable = create_json_chat_agent(self.chat_model, self.tools, self.prompt)
        self.tool_executor = ToolExecutor(self.tools)

        # Build and compile the workflow graph once.
        self.compiled_graph = self.build_workflow().compile()

    def _build_python_repl_tool(self):
        """
        Define and return a python_repl tool that executes code via the PythonREPL.
        """
        @tool
        def python_repl(code: str) -> str:
            """
            Execute the provided Python code and return its output.
            """
            try:
                result = self.repl.run(code)
                return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
            except Exception as e:
                return f"Error executing code: {str(e)}"
        return python_repl

    def run_agent(self, data: dict) -> dict:
        """
        Invoke the chat agent and return its outcome.
        Raises an exception if the agent call fails.
        """
        try:
            outcome = self.agent_runnable.invoke(data)
            return {"agent_outcome": outcome}
        except Exception as e:
            raise Exception(f"Error in run_agent: {str(e)}")  # Re-raise to halt workflow.

    def execute_tools(self, data: dict) -> dict:
        """
        Execute the tool action provided by the agent.
        This function automatically executes the tool action without user confirmation.
        """
        agent_action = data.get("agent_outcome")
        # Validate that the agent_action has the expected attributes.
        if not hasattr(agent_action, "tool") or not hasattr(agent_action, "tool_input"):
            raise ValueError("Agent action missing tool information.")
        output = self.tool_executor.invoke(agent_action)
        if "intermediate_steps" not in data:
            data["intermediate_steps"] = []
        data["intermediate_steps"].append((agent_action, str(output)))
        return data

    def should_continue(self, data: dict) -> str:
        """
        Decide whether to continue the loop or finish.
        Returns "end" if the agent outcome is an AgentFinish instance; otherwise, "continue".
        """
        if isinstance(data.get("agent_outcome"), AgentFinish):
            return "end"
        else:
            return "continue"

    def build_workflow(self) -> StateGraph:
        """
        Build and return the workflow graph.
        The graph contains two nodes:
          - "agent": runs the chat agent.
          - "action": executes the tool action.
        The graph loops until the agent finishes.
        """
        graph = StateGraph(AgentLoopState)
        graph.add_node("agent", self.run_agent)
        graph.add_node("action", self.execute_tools)
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self.should_continue,
            {"continue": "action", "end": END}
        )
        graph.add_edge("action", "agent")
        return graph

    def run_workflow(self, data: dict) -> dict:
        """
        Run the workflow until completion using the precompiled graph.
        Returns the final state.
        """
        final_state = None
        for state in self.compiled_graph.stream(data):
            final_state = state
        return final_state

# --- Main Execution for Testing ---
if __name__ == "__main__":
    # Load API key from .env
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not set in environment.")
    # Define initial input.
    inputs = {"input": "what is the weather in taiwan", "chat_history": []}
    # Instantiate the workflow with the provided API key and model.
    loop_workflow = AgentLoopWorkflow(openai_api_key=api_key, model="gpt-4")
    result = loop_workflow.run_workflow(inputs)
    # Print the final output if the workflow ended properly.
    if isinstance(result.get("agent_outcome"), AgentFinish):
        print(result["agent_outcome"].return_values["output"])
    else:
        print("Workflow did not finish properly.")
