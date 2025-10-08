import logging
import asyncio
from typing import List, Optional, Dict, Any, TypedDict

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field

from src.core_rag_engine import CoreRAGEngine
from src.stock import fetch_stock_news_documents
from src.scraper import scrape_urls_as_documents
from src.config import settings as app_settings

try:
    from langchain_tavily import TavilySearch

    TAVILY_LANGCHAIN_AVAILABLE = True
except ImportError:
    TAVILY_LANGCHAIN_AVAILABLE = False


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
    reflection_notes: List[str]  # Agent's thoughts about progress
    failed_steps: List[tuple]     # Track failures for retry with different approach
    max_iterations: int           # Prevent infinite loops
    current_iteration: int


class AgentLoopWorkflow:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        core_rag_engine_instance: Optional[CoreRAGEngine] = None,
        enable_tavily_search: Optional[bool] = None,
        enable_python_repl: Optional[bool] = None,
        custom_external_tools: Optional[List[Tool]] = None,
    ):
        self.openai_api_key = openai_api_key or app_settings.api.openai_api_key
        self.model_name = model or app_settings.agent.agent_model_name
        _enable_tavily = (
            enable_tavily_search if enable_tavily_search is not None else app_settings.agent.enable_tavily_search_by_default
        )
        _enable_repl = (
            enable_python_repl if enable_python_repl is not None else app_settings.agent.enable_python_repl_by_default
        )

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
                if TAVILY_LANGCHAIN_AVAILABLE:
                    self.tools.append(TavilySearch(api_key=tavily_key, max_results=3))
                    self.logger.info("Using langchain-tavily TavilySearch")
                else:
                    try:
                        tavily_tool = TavilySearchResults(max_results=3)
                        # Create a proper Tool wrapper
                        self.tools.append(
                            Tool(
                                name="TavilySearch",
                                func=tavily_tool.run,
                                description="Search the web for current information",
                            )
                        )
                        self.logger.info("Using langchain-community TavilySearchResults (langchain-tavily not available)")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize Tavily fallback: {e}")
            else:
                self.logger.warning("Tavily enabled but no API key provided")
        if _enable_repl:
            self.repl = PythonREPL()
            self.tools.append(
                Tool(
                    name="python_repl",
                    func=self.repl.run,
                    description="Executes Python code.",
                )
            )
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
                description="Answers questions using the adaptive RAG workflow.",
            ),
            Tool(
                name="InsightEngineIngest",
                func=self.core_rag_engine.ingest,
                description="Ingests documents into a collection.",
            ),
        ]

    def _create_data_feed_tools(self) -> List[Tool]:
        return [
            Tool(
                name="FetchStockNews",
                func=fetch_stock_news_documents,
                description="Fetches recent stock news articles.",
            ),
            Tool(
                name="ScrapeWebURLs",
                func=scrape_urls_as_documents,
                description="Scrapes content from web URLs.",
            ),
        ]

    def _get_planner_prompt(self) -> ChatPromptTemplate:
        """Creates and returns the planner prompt template."""
        parser = PydanticOutputParser(pydantic_object=Plan)

        # Format tool descriptions
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])

        # Define the prompt template
        prompt_template = (
            "You are an intelligent planner that creates step-by-step execution plans to achieve user goals.\n\n"
            "Available tools:\n{tool_descriptions}\n\n"
            "User Goal: {goal}\n\n"
            "Create a detailed plan to achieve this goal using ONLY the available tools listed above.\n"
            "Each step should specify:\n"
            "1. Which tool to use\n"
            "2. The exact input for that tool\n"
            "3. The reasoning for why this step is necessary\n\n"
            "Respond with a JSON object that follows this exact schema:\n"
            "{format_instructions}"
        )

        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "tool_descriptions": tool_descriptions,
            },
        )
        return prompt

    def plan_step(self, state: AgentLoopState) -> dict:
        self.logger.info("PLANNING: Generating execution plan...")

        prompt = self._get_planner_prompt()
        planner_chain = prompt | self.chat_model | PydanticOutputParser(pydantic_object=Plan)

        try:
            plan = planner_chain.invoke({"goal": state["input"]})
            return {"plan": plan, "current_step_index": 0}
        except Exception as e:
            self.logger.error(f"Failed to generate plan: {e}")
            return {"error": f"Failed to generate a plan: {e}"}

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

        new_past_step = (
            AgentAction(tool=tool_name, tool_input=tool_input, log=""),
            observation,
        )

        return {
            "past_steps": state["past_steps"] + [new_past_step],
            "current_step_index": state["current_step_index"] + 1,
        }

    def should_continue(self, state: AgentLoopState) -> str:
        if state["current_step_index"] >= len(state["plan"].steps):
            return "end"
        return "continue"

    def summarize_step(self, state: AgentLoopState) -> dict:
        # Dummy summary for now
        self.logger.info("SUMMARIZING results...")
        final_summary = "Plan executed. Summary of results: " + " ".join([str(s[1]) for s in state["past_steps"]])
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
            {"continue": "execute_step", "end": "summarize_step"},
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
            "error": None,
        }
        final_state = self.compiled_graph.invoke(initial_state)
        final_state["agent_outcome"] = AgentFinish(return_values={"output": final_state["final_summary"]}, log="")
        return final_state
    
def reflect_on_progress(self, state: AgentLoopState) -> dict:
    """Agent reflects on whether it's making progress toward goal."""
    reflection_prompt = f"""
    Original Goal: {state['input']}
    
    Steps taken so far:
    {self._format_past_steps(state['past_steps'])}
    
    Analyze:
    1. Are we making progress toward the goal?
    2. Should we continue with the current plan?
    3. Do we need to revise our approach?
    
    Respond with JSON: {{"continue": bool, "reasoning": str, "suggested_changes": str}}
    """
    
    reflection = self.chat_model.invoke(reflection_prompt)
    return {"reflection_notes": state["reflection_notes"] + [reflection]}

# Integrate with your existing memory_server.py
async def _load_past_context(self, state: AgentLoopState) -> dict:
    """Load relevant past conversations for context."""
    # Use the MCP memory server you already have!
    if self.mcp_memory_tool:
        relevant_memories = await self.mcp_memory_tool.retrieve_relevant_memories(
            query=state['input'],
            top_k=3
        )
        return {"retrieved_memories": relevant_memories}
    return {}

async def _store_execution_memory(self, state: AgentLoopState):
    """Store this execution for future reference."""
    if self.mcp_memory_tool:
        await self.mcp_memory_tool.store_conversation_context(
            session_id=self._generate_session_id(),
            question=state['input'],
            answer=state['final_summary'],
            context_docs=state['past_steps']
        )

async def execute_parallel_steps(self, state: AgentLoopState) -> dict:
    """Execute independent steps in parallel for speed."""
    plan_steps = state['plan'].steps
    current_idx = state['current_step_index']
    
    # Identify independent steps (no dependencies)
    parallel_steps = self._find_parallel_steps(plan_steps, current_idx)
    
    # Execute in parallel
    tasks = [
        self._execute_single_step(step) 
        for step in parallel_steps
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Update state with all results
    new_past_steps = state['past_steps'] + list(zip(parallel_steps, results))
    return {
        "past_steps": new_past_steps,
        "current_step_index": current_idx + len(parallel_steps)
    }