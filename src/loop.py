import os
import logging
import tempfile
from typing import List, Optional, Dict, Any, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langgraph.prebuilt import ToolNode
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
    """
    State dictionary for the plan-and-execute agent loop with memory.
    """
    input: str
    plan: Plan
    past_steps: List[tuple]
    current_step: int
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
            self.logger.error("OPENAI_API_KEY missing for AgentLoopWorkflow's ChatOpenAI.")

        self.chat_model = ChatOpenAI(model=self.model_name, openai_api_key=self.openai_api_key)

        self.tools: List[Tool] = []

        if _enable_tavily:
            tavily_key_from_config = app_settings.api.tavily_api_key
            if tavily_key_from_config:
                self.tools.append(TavilySearchResults(api_key=tavily_key_from_config, max_results=3))
            else:
                self.logger.warning("TAVILY_API_KEY not set in config; TavilySearchResults tool disabled for agent.")

        if _enable_repl:
            self.repl = PythonREPL()
            self.tools.append(self._build_python_repl_tool())

        if self.core_rag_engine:
            self.tools.extend(self._create_core_rag_engine_tools())

        self.tools.extend(self._create_data_feed_tools())

        if custom_external_tools:
            self.tools.extend(custom_external_tools)

        self.logger.info(f"Agent initialized with {len(self.tools)} tools.")

        self.memory_store = None
        self.memory_retriever = None
        try:
            memory_persist_dir = os.path.join(tempfile.gettempdir(), "agent_memory_store")
            os.makedirs(memory_persist_dir, exist_ok=True)
            
            embedding_model = self.core_rag_engine.embedding_model if self.core_rag_engine else OpenAIEmbeddings()

            self.memory_store = Chroma(
                collection_name="agent_long_term_memory",
                embedding_function=embedding_model,
                persist_directory=memory_persist_dir
            )
            self.memory_retriever = self.memory_store.as_retriever(search_kwargs={'k': 3})
            self.logger.info(f"Long-term memory initialized at {memory_persist_dir}")

        except Exception as e:
            self.logger.error(f"Failed to initialize long-term memory: {e}", exc_info=True)

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

        tools.append(Tool(name="FetchStockNews",
                          func=fetch_stock_news_documents,
                          description=("Fetch recent stock news articles. "
                                       "Input: {'tickers_input': str or List[str], 'max_articles_per_ticker': int}. "
                                       "Returns List[Document].")
                       )
                    )

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
    
    def _create_planner_chain(self) -> LLMChain:
        """Creates a chain that generates a plan from a user goal and past memories."""
        parser = PydanticOutputParser(pydantic_object=Plan)
        tool_descriptions = "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)

        prompt_template = (
            "You are an expert planner. Your task is to create a step-by-step plan to achieve the user's goal using the available tools.\n"
            "If you have been provided with summaries of relevant past tasks, use them to inform your plan.\n\n"
            "Relevant Past Task Summaries (Memories):\n---\n{retrieved_memories}\n---\n\n"
            "Available Tools:\n{tool_descriptions}\n\n"
            "Respond with a JSON object matching this schema:\n{format_instructions}\n\n"
            "User Goal: {goal}\n\n"
            "Provide your JSON plan:"
        )
        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "tool_descriptions": tool_descriptions,
            }
        )
        
        json_llm = self.core_rag_engine.json_llm if self.core_rag_engine else self.chat_model
        
        chain = LLMChain(llm=json_llm, prompt=prompt, output_parser=parser)

        return chain
    
    def _create_reflection_chain(self) -> LLMChain:
        """Creates a chain that reflects on a completed plan step."""
        prompt = ChatPromptTemplate.from_template(
            "You are an expert agent. You just completed a step in your plan.\n"
            "Goal: {goal}\n"
            "Plan Step: {step_reasoning}\n"
            "Tool Used: {tool_name}\n"
            "Tool Output: {observation}\n\n"
            "Based on this, what is the key takeaway or result from this step? Be concise (1-2 sentences)."
        )
        return LLMChain(llm=self.chat_model, prompt=prompt, output_parser=StrOutputParser())

    def reflection_step(self, state: AgentLoopState) -> AgentLoopState:
        """Reflects on the last executed step and adds to the scratchpad."""
        self.logger.info("REFLECTING on last step...")
        reflection_chain = self._create_reflection_chain()

        # Get the last executed step details
        step_index = state["current_step"] - 1 # execute_step increments it
        plan_step = state["plan"].steps[step_index - 1]
        action, observation = state["past_steps"][-1]

        try:
            reflection = reflection_chain.invoke({
                "goal": state["input"],
                "step_reasoning": plan_step.reasoning,
                "tool_name": plan_step.tool,
                "observation": observation,
            })
            state.setdefault("scratchpad", []).append(reflection)
            self.logger.info(f"REFLECTION: '{reflection}'")
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}", exc_info=True)
            state.setdefault("scratchpad", []).append(f"Error reflecting on step {step_index}: {e}")

        return state

    def _create_summary_chain(self) -> LLMChain:
        """Creates a chain that generates a final summary from the scratchpad."""
        prompt = ChatPromptTemplate.from_template(
            "You are an expert agent. You have completed your plan successfully.\n"
            "Your original goal was: {goal}\n"
            "You have completed a series of steps and recorded your key takeaways in a scratchpad.\n\n"
            "Scratchpad:\n---\n{scratchpad_content}\n---\n\n"
            "Based on your original goal and the scratchpad, provide a final, comprehensive answer for the user."
        )
        return LLMChain(llm=self.chat_model, prompt=prompt, output_parser=StrOutputParser())

    def summarize_step(self, state: AgentLoopState) -> AgentLoopState:
        """Generates a final summary from the scratchpad contents."""
        self.logger.info("SUMMARIZING results...")
        summary_chain = self._create_summary_chain()
        
        scratchpad_content = "\n".join(f"- {s}" for s in state.get("scratchpad", []))
        
        try:
            summary = summary_chain.invoke({
                "goal": state["input"],
                "scratchpad_content": scratchpad_content,
            })
            state["final_summary"] = summary
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}", exc_info=True)
            state["final_summary"] = f"Error generating final summary: {e}"
            
        return state
    
    def _create_task_summary_chain(self) -> LLMChain:
        """Creates a chain that summarizes a completed task for long-term memory."""
        prompt = ChatPromptTemplate.from_template(
            "You are an expert agent. You just completed a task. Your goal was: '{goal}'.\n"
            "You generated the following final summary: '{final_summary}'.\n\n"
            "Create a concise summary of this completed task to store in your long-term memory. "
            "This memory should be useful for planning similar tasks in the future. "
            "Focus on the goal and the key outcome. (1-3 sentences)."
        )
        return LLMChain(llm=self.chat_model, prompt=prompt, output_parser=StrOutputParser())

    def save_memory_step(self, state: AgentLoopState) -> AgentLoopState:
        """Summarizes the completed task and saves it to the long-term memory store."""
        if not self.memory_store or state.get("error"):
            self.logger.warning("Skipping memory storage due to error or uninitialized store.")
            return state

        self.logger.info("SAVING TO MEMORY: Creating task summary...")
        task_summary_chain = self._create_task_summary_chain()
        try:
            memory_text = task_summary_chain.invoke({
                "goal": state["input"],
                "final_summary": state["final_summary"]
            })
            
            # Save the summary as a Document
            memory_doc = Document(page_content=memory_text)
            self.memory_store.add_documents([memory_doc])
            self.logger.info(f"SAVED TO MEMORY: '{memory_text}'")
        except Exception as e:
            self.logger.error(f"Failed to save task summary to memory: {e}", exc_info=True)
            
        return state

    def plan_step(self, state: AgentLoopState) -> AgentLoopState:
        """Recalls memories and generates a plan."""
        self.logger.info("PLANNING: Recalling long-term memories...")
        
        # Recall relevant memories
        retrieved_memories = "No relevant memories found."
        if self.memory_retriever:
            try:
                retrieved_docs = self.memory_retriever.get_relevant_documents(state["input"])
                if retrieved_docs:
                    retrieved_memories = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    self.logger.info(f"Recalled {len(retrieved_docs)} memories.")
            except Exception as e:
                self.logger.error(f"Memory retrieval failed: {e}", exc_info=True)
        
        state["retrieved_memories"] = retrieved_memories

        self.logger.info("PLANNING: Generating execution plan...")
        planner_chain = self._create_planner_chain()
        try:
            plan = planner_chain.invoke({
                "goal": state["input"],
                "retrieved_memories": retrieved_memories
            })
            state["plan"] = plan
            state["current_step"] = 1
            self.logger.info(f"PLANNING: Plan generated with {len(plan.steps)} steps.")
        except Exception as e:
            self.logger.error(f"PLANNING: Failed to generate plan. Error: {e}", exc_info=True)
            state["error"] = f"Failed to generate a plan: {e}"
        return state
    
    def execute_tool_step(self, state: AgentLoopState) -> dict:
        """Executes the tool for the current plan step and returns the output."""
        plan_step = state["plan"].steps[state["current_step"] - 1]
        tool_name = plan_step.tool
        tool_input = plan_step.tool_input
    
        tool_map = {tool.name: tool for tool in self.tools}
        if tool_name in tool_map:
            try:
                tool_result = tool_map[tool_name].invoke(tool_input)
                return {
                    "past_steps": [(tool_name, str(tool_result))],
                    "error": None,
                }
            except Exception as e:
                return {
                    "past_steps": [(tool_name, f"Error: {e}")],
                    "error": str(e),
                }
        else:
            return {
                "past_steps": [(tool_name, "Error: Tool not found.")],
                "error": f"Tool '{tool_name}' not found.",
            }

    
    def build_workflow(self) -> StateGraph:
    """Builds the plan-reflect-execute workflow."""
    graph = StateGraph(AgentLoopState)

    graph.add_node("plan_step", self.plan_step)
    graph.add_node("execute_tool_step", self.execute_tool_step)
    graph.add_node("reflection_step", self.reflection_step)
    graph.add_node("summarize_step", self.summarize_step)
    graph.add_node("save_memory_step", self.save_memory_step)

    graph.set_entry_point("plan_step")

    graph.add_edge("plan_step", "execute_tool_step")
    graph.add_edge("execute_tool_step", "reflection_step")

    graph.add_conditional_edges(
        "reflection_step",
        self.should_continue_planned_workflow,
        {
            "continue": "execute_tool_step",
            "end": "summarize_step"
        },
    )
    graph.add_edge("summarize_step", "save_memory_step")
    graph.add_edge("save_memory_step", END)

    return graph.compile()

    def _get_current_task(self, state: AgentLoopState) -> dict:
        """Helper to get the current agent action from the plan."""
        step = state['current_step'] - 1
        if step < 0 or step >= len(state.get("plan").steps):
            return END
        plan_step = state["plan"].steps[step]
        action = AgentAction(tool=plan_step.tool, tool_input=plan_step.tool_input, log="")
        return {"messages": [action]}

    def should_continue_planned_workflow(self, state: AgentLoopState) -> str:
        """Determines if the planned execution should continue."""
        if state.get("error"):
            self.logger.error(f"Workflow ending due to error: {state['error']}")
            return END
        
        # The 'current_step' has been incremented by execute_step already
        last_step_executed = state["current_step"] - 1
        if last_step_executed >= len(state["plan"].steps):
            self.logger.info("Workflow finished: All plan steps executed. Proceeding to summary.")
            return "end" # Signal to go to summarize_step
        
        return "continue"
    
    def run_workflow(self, goal: str) -> AgentLoopState:
        """
        Public method: runs the agent on a high-level goal by planning and executing.
        Returns the final state containing the plan, scratchpad, and final summary.
        """
        initial_state: AgentLoopState = {
            "input": goal,
            "plan": None,
            "past_steps": [],
            "current_step": 0,
            "scratchpad": [],
            "final_summary": "",
            "retrieved_memories": None,
            "error": None
        }
        
        final_state = self.compiled_graph.invoke(initial_state)

        # Structure the final output for compatibility
        if not final_state.get("error"):
            final_output = {
                "output": final_state.get("final_summary", "Agent task finished without a summary."),
                "scratchpad": final_state.get("scratchpad", []),
                "recalled_memories": final_state.get("retrieved_memories", "None")
            }
            final_state["agent_outcome"] = AgentFinish(return_values=final_output, log="")
        else:
            # +++ ADD THIS ELSE BLOCK FOR ROBUST ERROR HANDLING +++
            error_message = final_state.get('error', 'Unknown error')
            final_output = {
                "output": f"Agent task failed with error: {error_message}"
            }
            final_state["agent_outcome"] = AgentFinish(return_values=final_output, log=f"Error: {error_message}")
        
        
        return final_state
    
