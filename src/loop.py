import os
import logging
from typing import List, Optional, Dict, Any

from langchain.agents import create_json_chat_agent, AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import StateGraph, END

from src.core_rag_engine import CoreRAGEngine
from src.stock import fetch_stock_news_documents
from src.scraper import scrape_urls_as_documents


class AgentLoopState(Dict[str, Any]):
    """
    State dictionary for the agent loop.
    Keys:
      - input: str (the user's high-level goal)
      - chat_history: List[BaseMessage]
      - agent_outcome: AgentAction | AgentFinish | None
      - intermediate_steps: List[ tuple(AgentAction, str) ]
    """
    pass


class AgentLoopWorkflow:
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o",
        core_rag_engine_instance: Optional[CoreRAGEngine] = None,
        enable_tavily_search: bool = True,
        enable_python_repl: bool = False,
        custom_external_tools: Optional[List[Tool]] = None
    ):
        self.openai_api_key = openai_api_key
        self.model_name = model
        self.core_rag_engine = core_rag_engine_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize the chat model
        self.chat_model = ChatOpenAI(model=self.model_name openai_api_key=self.openai_api_key)

        # Build tool list
        self.tools: List[Tool] = []

        # Tavily search
        if enable_tavily_search:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                self.tools.append(TavilySearchResults(api_key=tavily_key, max_results=3))
            else:
                self.logger.warning("TAVILY_API_KEY not set; TavilySearchResults tool disabled.")

        # Python REPL
        if enable_python_repl:
            self.repl = PythonREPL()
            self.tools.append(self._build_python_repl_tool())

        # CoreRAGEngine tools
        if self.core_rag_engine:
            self.tools.extend(self._create_core_rag_engine_tools())

        # Data feed tools
        self.tools.extend(self._create_data_feed_tools())

        # Custom external tools
        if custom_external_tools:
            self.tools.extend(custom_external_tools)

        self.logger.info(f"Agent initialized with {len(self.tools)} tools.")

        # System prompt guiding multi-step usage
        system_message = """ You are the InsightEngine Agent. Your job is to fulfill complex tasks by planning and using tools intelligently. For data acquisition:
        - Use FetchStockNews and ScrapeWebURLs to retrieve raw documents.
        - Then use InsightEngineIngest to add them into a named collection.
        For reasoning:
        - Use InsightEngineRAGWorkflow with a question and collection_name to get answers from ingested content.
        Always think step-by-step:
        1. Plan which tool to call.
        2. Execute the tool.
        3. Observe output, then plan next step.
        Provide final answer when finished.
        """

        self.prompt = ChatPromptTemplate(
            input_variables=["input", "chat_history", "tool_names", "agent_scratchpad"],
            messages=[
                SystemMessage(content=system_message),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                HumanMessage(content=(
                    "TOOLS\n------\n"
                    "Available tools: {tool_names}\n\n"
                    "When you call a tool, respond with a JSON object "
                    "with 'action' and 'action_input'.\n\n"
                    "USER GOAL\n----------\n{input}"
                )),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )

        # Create the agent and executor
        self.agent_runnable = create_json_chat_agent(
            llm=self.chat_model,
            tools=self.tools,
            prompt=self.prompt
        )
        self.tool_executor = ToolExecutor(self.tools)

        # Compile the LangGraph workflow
        self.compiled_graph = self.build_workflow().compile()

    def _build_python_repl_tool(self) -> Tool:
        return Tool(
            name="python_repl",
            func=self.repl.run,
            description="Executes Python code and returns stdout."
        )

    def _create_core_rag_engine_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        # Run RAG workflow tool
        tools.append(Tool(
            name="InsightEngineRAGWorkflow",
            func=self.core_rag_engine.run_full_rag_workflow,
            description=(
                "Answer questions using the adaptive RAG workflow. "
                "Input: {'question': str, 'collection_name': Optional[str]}. "
                "Returns a dict with 'answer' and 'sources'."
            )
        ))

        # Ingest tool
        tools.append(Tool(
            name="InsightEngineIngest",
            func=self.core_rag_engine.ingest,
            description=(
                "Ingest documents or sources into a collection. "
                "Input: { 'sources': List[{'type':str,'value':Any}], "
                "'direct_documents': Optional[List[Document]], "
                "'collection_name': Optional[str], "
                "'recreate_collection': Optional[bool] }."
            )
        ))

        self.logger.info(f"Created {len(tools)} CoreRAGEngine tools.")
        return tools

    def _create_data_feed_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        tools.append(Tool(
            name="FetchStockNews",
            func=fetch_stock_news_documents,
            description=(
                "Fetch recent stock news articles. "
                "Input: {'tickers_input': str or List[str], 'max_articles_per_ticker': int}. "
                "Returns List[Document]."
            )
        ))

        tools.append(Tool(
            name="ScrapeWebURLs",
            func=scrape_urls_as_documents,
            description=(
                "Scrape content from given web URLs. "
                "Input: {'urls': List[str], 'user_goal_for_scraping': Optional[str]}. "
                "Returns List[Document]."
            )
        ))

        self.logger.info(f"Created {len(tools)} data feed tools.")
        return tools

    def run_agent(self, state: AgentLoopState) -> AgentLoopState:
        outcome = self.agent_runnable.invoke({
            "input": state["input"],
            "chat_history": state.get("chat_history", []),
            "agent_scratchpad": []
        })
        state["agent_outcome"] = outcome
        return state

    def execute_tools(self, state: AgentLoopState) -> AgentLoopState:
        action: AgentAction = state["agent_outcome"]
        result = self.tool_executor.invoke(action)
        state.setdefault("intermediate_steps", []).append((action, str(result)))
        return state

    def should_continue(self, state: AgentLoopState) -> str:
        return "end" if isinstance(state.get("agent_outcome"), AgentFinish) else "continue"

    def build_workflow(self) -> StateGraph:
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

    def run_workflow(self, goal: str) -> AgentLoopState:
        """
        Public method: run the agent on a high-level goal.
        Returns the final state containing intermediate_steps and agent_outcome.
        """
        initial_state: AgentLoopState = {
            "input": goal,
            "chat_history": [],
            "agent_outcome": None,
            "intermediate_steps": []
        }
        final_state = None
        for st in self.compiled_graph.stream(initial_state):
            final_state = st
        return final_state
