
"""
stock.py: Stock analysis agent workflow using LangGraph.

Fetches stock news from Yahoo Finance, analyzes it, and suggests trading ideas.
LangGraph coordinates agents; Streamlit provides the interface.
"""

import os
import json
from typing import List, TypedDict, Sequence, Optional
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.graph import StateGraph, END

load_dotenv()

class AgentState(TypedDict):
    """
    State passed between graph nodes.
    """
    messages: Sequence[BaseMessage]
    next: str
    error: Optional[str]

def get_valid_tickers(user_input: str) -> List[str]:
    """
    Extract stock tickers from user input.
    1. Normalize separators to commas.
    2. Split, uppercase, strip.
    3. Keep tokens 1–12 chars, allowed [A–Z0–9.-], must have a letter.
    4. Return sorted, unique list.
    """
    if not user_input:
        return []

    normalized = user_input.replace(';', ',').replace(' ', ',')
    raw_tokens = normalized.split(',')

    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    valid: List[str] = []

    for tok in raw_tokens:
        tok = tok.strip().upper()
        
        if tok:
            if (all(ch in allowed_chars for ch in tok)):
                if (any(ch.isalpha() for ch in tok)):
                    if (tok not in valid):
                        valid.append(tok)
        
    valid.sort()
    return valid


class StockWorkflow:
    RESEARCHER  = "Stock_Researcher"
    ANALYZER    = "Market_Analyst"
    RECOMMENDER = "Trade_Recommender"
    SUPERVISOR  = "Workflow_Supervisor"
    FINISH      = "END_WORKFLOW"

    def __init__(self,
                 model_name: str,
                 api_key: str,
                 verbose: bool = False,
                 ui_logger=None):
                     
        self.ui_logger = ui_logger
        self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0, verbose=verbose)
        self.tool_instance = YahooFinanceNewsTool()

        @tool("Stock_News_Research_Tool")
        def research_tool(tickers: str) -> str:
            self._log(f"Fetching news for: {tickers}")
            
            try:
                news = self.tool_instance.run(tickers)
            except Exception as e:
                msg = f"Error fetching news for {tickers}: {e}"
                self._log(msg, level="error")
                return msg
            
            if not news or "Cannot find any article" in news:
                msg = f"No news found for: {tickers}"
                self._log(msg, level="warning")
                return msg
            
            return news

        self.tools = [research_tool]
        self._prepare_prompts()

        self.researcher = AgentExecutor(agent=create_openai_tools_agent(self.llm, self.tools, self.researcher_prompt),
                                        tools=self.tools,
                                        verbose=verbose)
                     
        self.analyzer = AgentExecutor(agent=create_openai_tools_agent(self.llm, [], self.analyzer_prompt),
                                      tools=[],
                                      verbose=verbose)
                     
        self.recommender = AgentExecutor(agent=create_openai_tools_agent(self.llm, [], self.recommender_prompt),
                                         tools=[],
                                         verbose=verbose)

        self.routing_function = {
            "name": "route_to_next",
            "description": "Choose the next agent or finish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "next_step": {
                        "enum": [
                            self.FINISH,
                            self.RESEARCHER,
                            self.ANALYZER,
                            self.RECOMMENDER]
                    }
                },
                "required": ["next_step"]
            }
        }

    def _prepare_prompts(self):
        """Define prompts for each agent and the supervisor."""
        self.researcher_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Stock Researcher. Use the Stock_News_Research_Tool to fetch news."),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")])
        
        self.analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Market Analyst. Identify trends, risks, and opportunities."),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")])
        
        self.recommender_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Trade Recommender. Suggest trading ideas based on analysis."),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")])

        finish_token = self.FINISH
            self.supervisor_system = (f"You are the Workflow Supervisor. "
                                      f"Agents: {self.RESEARCHER}, {self.ANALYZER}, {self.RECOMMENDER}. "
                                      f"If research failed or an error occurred, or when done, route to {finish_token}.")
        
        options = ", ".join([finish_token, self.RESEARCHER, self.ANALYZER, self.RECOMMENDER])
        self.supervisor_options = f"Options: {options}"

    def _log(self, message: str, level: str = "info"):
        """Log to Streamlit UI or print to console."""
        if self.ui_logger:
            getattr(self.ui_logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")

    def _run_node(self, state: AgentState, executor: AgentExecutor, node_name: str) -> dict:
        self._log(f"Running {node_name}")
        
        try:
            result = executor.invoke({"messages": state["messages"]})
            output = result.get("output", "")
        except Exception as exc:
            msg = f"{node_name} exception: {exc}"
            self._log(msg, level="error")
            return {"messages": [HumanMessage(content=msg, name="SystemError")], "error": msg}

        for kw in ("Error fetching news", "No news found"):
            if kw in output:
                self._log(f"{node_name} issue: {output}", level="warning")
                return {"messages": [HumanMessage(content=output, name=node_name)], "error": output}

        self._log(f"{node_name} completed")
        return {"messages": [HumanMessage(content=output, name=node_name)], "error": None}

    def run_researcher(self, state: AgentState) -> dict:
        return self._run_node(state, self.researcher, self.RESEARCHER)

    def run_analyzer(self, state: AgentState) -> dict:
        return self._run_node(state, self.analyzer, self.ANALYZER)

    def run_recommender(self, state: AgentState) -> dict:
        return self._run_node(state, self.recommender, self.RECOMMENDER)

    def run_supervisor(self, state: AgentState) -> dict:
        if state.get("error"):
            err = state["error"]
            self._log(f"Supervisor ending due to error: {err}", level="warning")
            return {"next": self.FINISH, "error": err}

        messages = [SystemMessage(content=self.supervisor_system)] + state["messages"] + [SystemMessage(content=self.supervisor_options)]

        try:
            response = self.llm(messages=messages,
                                functions=[self.routing_function],
                                function_call={"name": "route_to_next"})
            
            fn_call = response.additional_kwargs.get("function_call", {})
            args = fn_call.get("arguments", "{}")
            parsed = json.loads(args)
            next_step = parsed.get("next_step", self.FINISH)
        
        except Exception as exc:
            err = f"Supervisor exception: {exc}"
            self._log(err, level="error")
            return {"next": self.FINISH, "error": err}

        self._log(f"Supervisor routed to: {next_step}")
        return {"next": next_step, "error": None}

    def compile_graph(self) -> 'Runnable':
        """
        Build and return the LangGraph workflow.
        """
        graph = StateGraph(AgentState)
        graph.add_node(self.RESEARCHER, self.run_researcher)
        graph.add_node(self.ANALYZER, self.run_analyzer)
        graph.add_node(self.RECOMMENDER, self.run_recommender)
        graph.add_node(self.SUPERVISOR, self.run_supervisor)

        graph.add_edge(self.RESEARCHER, self.SUPERVISOR)
        graph.add_edge(self.ANALYZER, self.SUPERVISOR)
        graph.add_edge(self.RECOMMENDER, self.SUPERVISOR)

        def get_next(state: AgentState) -> str:
            return state["next"]

        routing = {self.RESEARCHER: self.RESEARCHER,
                   self.ANALYZER: self.ANALYZER,
                   self.RECOMMENDER: self.RECOMMENDER,
                   self.FINISH: END}
        
        graph.add_conditional_edges(self.SUPERVISOR, get_next, routing)
        graph.set_entry_point(self.SUPERVISOR)

        return graph.compile()


def run_stock_analysis_ui():
    """
    Streamlit UI:
      - Sidebar: model, API key, verbosity
      - Main: ticker input, logs, run button
    """
    st.set_page_config(page_title="Stock News Analysis", layout="wide")
    st.title("Stock News Analysis Workflow")
    st.sidebar.header("Configuration")

    model_name = st.sidebar.selectbox("OpenAI Model", 
                                      ["gpt-4-turbo", "gpt-4-turbo-preview", "gpt-3.5-turbo"])
    
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    verbose = st.sidebar.checkbox("Enable Console Verbose Logging", value=False)

    tickers_input = st.text_input("Enter stock tickers (comma or space separated):")
    log_area = st.container()

    if st.button("Run Analysis"):
        log_area.empty()
        log_area.info("Starting workflow")

        if not api_key:
            log_area.warning("Please enter your OpenAI API key.")
            st.stop()

        if not tickers_input.strip():
            log_area.warning("Please enter at least one ticker.")
            st.stop()

        tickers = get_valid_tickers(tickers_input)
        if not tickers:
            log_area.warning("No valid tickers found. Use symbols like MSFT or AAPL.")
            st.stop()

        query = ",".join(tickers)
        log_area.success(f"Tickers to process: {query}")

        try:
            wf = StockWorkflow(model_name, api_key, verbose, ui_logger=log_area)
            graph = wf.compile_graph()
            state = {"messages": [HumanMessage(content=f"Research news for: {query}", name=wf.RESEARCHER)],
                     "error": None}

            final_state = None
            for step in graph.stream(state):
                final_state = list(step.values())[0]

            log_area.success("Workflow completed")
            st.subheader("Final Outcome")

            if final_state.get("error"):
                st.error(f"Workflow ended with error: {final_state['error']}")
            else:
                last = final_state["messages"][-1].content
                st.info(last)

        except Exception as e:
            log_area.error(f"Critical error: {e}")
            st.error(f"Critical error: {e}")


if __name__ == "__main__":
    run_stock_analysis_ui()
