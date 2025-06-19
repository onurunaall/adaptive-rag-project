import os
import logging
import tempfile
from typing import List, Optional, Dict, Any, TypedDict

from langchain_core.agents import AgentAction, AgentFinish
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field

from src.core_rag_engine import CoreRAGEngine
from src.stock import fetch_stock_news_documents
from src.scraper import scrape_urls_as_documents
from src.config import settings as app_settings


class PlanStep(BaseModel):
    """A single step in the execution plan."""
    tool: str = Field(description="The name of the tool to use for this step.")
    tool_input: Dict[str, Any] = Field(description="The dictionary input for the tool.")
    reasoning: str = Field(description="The reasoning behind choosing this tool and input.")

class Plan(BaseModel):
    """The complete, multi-step plan to achieve the user's goal."""
    steps: List[PlanStep] = Field(description="The list of sequential steps to execute.")

class AgentLoopState(TypedDict):
    """State dictionary for the plan-and-execute agent loop."""
    input: str
    plan: Plan
    past_steps: List[tuple]
    current_step_index: int
    scratchpad: List[str]
    final_summary: str
    retrieved_memories: Optional[str]
    error: Optional[str]


class AgentLoopWorkflow:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        core_rag_engine_instance: Optional[CoreRAGEngine] = None,
        enable_tavily_search: Optional[bool] = None,
        enable_python_repl: Optional[bool] = None,
        custom_external_tools: Optional[List[Tool]] = None
    ):
        self.openai_api_key = openai_api_key or app_settings.api.openai_api_key
        self.model_name = model or app_settings.agent.agent_model_name
        _enable_tavily = enable_tavily_search if enable_tavily_search is not None else app_settings.agent.enable_tavily_search_by_default
        _enable_repl = enable_python_repl if enable_python_repl is not None else app_settings.agent.enable_python_repl_by_default
        
        self.core_rag_engine = core_rag_engine_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.openai_api_key:
            self.logger.error("OPENAI_API_KEY missing for AgentLoopWorkflow.")
        
        self.chat_model = ChatOpenAI(model=self.model_name, openai_api_key=self.openai_api_key)
        
        self.tools: List[Tool] = []
        if _enable_tavily:
            tavily_key = app_settings.api.tavily_api_key
            if tavily_key:
                self.tools.append(TavilySearch(api_key=tavily_key, max_results=3))
        if _enable_repl:
            self.repl = PythonREPL()
            self.tools.append(Tool(name="python_repl", func=self.repl.run, description="Executes Python code."))
        if self.core_rag_engine:
            self.tools.extend(self._create_core_rag_engine_tools())
        self.tools.extend(self._create_data_feed_tools())
        if custom_external_tools:
            self.tools.extend(custom_external_tools)

        self.logger.info(f"Agent initialized with {len(self.tools)} tools.")
        self.compiled_graph = self.build_workflow()

    def _create_core_rag_engine_tools(self) -> List[Tool]:
        return [
            Tool(
                name="InsightEngineRAGWorkflow",
                func=self.core_rag_engine.run_full_rag_workflow,
                description="Answers questions using the adaptive RAG workflow."
            ),
            Tool(
                name="InsightEngineIngest",
                func=self.core_rag_engine.ingest,
                description="Ingests documents into a collection."
            )
        ]

    def _create_data_feed_tools(self) -> List[Tool]:
        return [
            Tool(
                name="FetchStockNews",
                func=fetch_stock_news_documents,
                description="Fetches recent stock news articles."
            ),
            Tool(
                name="ScrapeWebURLs",
                func=scrape_urls_as_documents,
                description="Scrapes content from web URLs."
            )
        ]

    def plan_step(self, state: AgentLoopState) -> dict:
        self.logger.info("PLANNING: Generating execution plan...")
        # This is a dummy planner for now. In a real scenario, this would call an LLM.
        # Based on the user's goal, it creates a plan.
        # For the test, we will mock this entire function's return value.
        plan = Plan(steps=[]) # Dummy plan
        return {"plan": plan, "current_step_index": 0}

    def execute_step(self, state: AgentLoopState) -> dict:
        self.logger.info(f"EXECUTING step {state['current_step_index'] + 1}...")
        plan_step = state["plan"].steps[state["current_step_index"]]
        tool_name = plan_step.tool
        tool_input = plan_step.tool_input
        
        selected_tool = next((t for t in self.tools if t.name == tool_name), None)
        if not selected_tool:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                observation = selected_tool.invoke(tool_input)
            except Exception as e:
                observation = f"Error executing tool '{tool_name}': {e}"

        new_past_step = (AgentAction(tool=tool_name, tool_input=tool_input, log=""), observation)
        
        return {"past_steps": state["past_steps"] + [new_past_step],
                "current_step_index": state["current_step_index"] + 1}

    def should_continue(self, state: AgentLoopState) -> str:
        if state["current_step_index"] >= len(state["plan"].steps):
            return "end"
        return "continue"
        
    def summarize_step(self, state: AgentLoopState) -> dict:
        # Dummy summary for now
        self.logger.info("SUMMARIZING results...")
        final_summary = "Plan executed. Summary of results: " + " ".join([str(s[1]) for s in state['past_steps']])
        return {"final_summary": final_summary}

    def build_workflow(self) -> StateGraph:
        graph = StateGraph(AgentLoopState)
        graph.add_node("plan_step", self.plan_step)
        graph.add_node("execute_step", self.execute_step)
        graph.add_node("summarize_step", self.summarize_step)
        
        graph.set_entry_point("plan_step")
        graph.add_edge("plan_step", "execute_step")
        graph.add_edge("summarize_step", END)

        graph.add_conditional_edges(
            "execute_step",
            self.should_continue,
            {"continue": "execute_step", "end": "summarize_step"}
        )
        
        return graph.compile()

    def run_workflow(self, goal: str) -> dict:
        initial_state: AgentLoopState = {
            "input": goal,
            "plan": None,
            "past_steps": [],
            "current_step_index": 0,
            "scratchpad": [],
            "final_summary": "",
            "retrieved_memories": None,
            "error": None
        }
        final_state = self.compiled_graph.invoke(initial_state)
        final_state["agent_outcome"] = AgentFinish(return_values={"output": final_state["final_summary"]}, log="")
        return final_state

