import os
from typing import List
import logging
import streamlit as st
import json

from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.core_rag_engine import CoreRAGEngine
from src.stock import fetch_stock_news_documents
from src.scraper import scrape_urls_as_documents
from src.loop import AgentLoopWorkflow, AgentLoopState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Initialize chat history
if "qa_chat_history" not in st.session_state:
    st.session_state.qa_chat_history: List[BaseMessage] = []

st.set_page_config(page_title="InsightEngine - Adaptive RAG", layout="wide")
st.title("InsightEngine – Adaptive RAG")


@st.cache_resource
def load_engine():
    return CoreRAGEngine()


engine = load_engine()

# Sidebar: Config & Ingest
st.sidebar.header("Config & Ingest")
collection_name = st.sidebar.text_input("Collection name", value=engine.default_collection_name)
recreate_collection = st.sidebar.checkbox("Recreate collection", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Add Sources")
uploaded = st.sidebar.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
urls_text = st.sidebar.text_area("URLs (one per line)", height=150)

if st.sidebar.button("Ingest Documents"):
    sources = []
    if uploaded:
        for f in uploaded:
            sources.append({"type": "uploaded_pdf", "value": f})
    for line in urls_text.splitlines():
        u = line.strip()
        if u:
            sources.append({"type": "url", "value": u})
    if not sources:
        st.sidebar.warning("No sources to ingest.")
    else:
        try:
            engine.ingest(
                sources=sources,
                collection_name=collection_name,
                recreate_collection=recreate_collection,
            )
            st.sidebar.success("Documents ingested.")
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")

# Stock News Feed
st.sidebar.markdown("---")
st.sidebar.subheader("Stock News Feed")

stock_tickers_input = st.sidebar.text_input(
    "Enter stock tickers (e.g., AAPL, MSFT, NVDA)",
    help="Comma or space-separated ticker symbols.",
)

stock_news_collection_name = st.sidebar.text_input(
    "Collection name for stock news",
    value="stock_news",
    help="Where stock news will be stored.",
)

recreate_stock_news_collection = st.sidebar.checkbox(
    "Recreate stock news collection if it exists",
    value=False,
    key="recreate_stock_news",
)

max_articles_stock = st.sidebar.number_input("Max articles per ticker", min_value=1, max_value=20, value=3, step=1)

if st.sidebar.button("Ingest Stock News"):
    if not stock_tickers_input.strip():
        st.sidebar.warning("Please enter at least one stock ticker.")
    else:
        with st.spinner(f"Fetching & ingesting news for {stock_tickers_input}..."):
            try:
                news_documents = fetch_stock_news_documents(
                    tickers_input=stock_tickers_input,
                    max_articles_per_ticker=max_articles_stock,
                )
                if news_documents:
                    engine.ingest(
                        direct_documents=news_documents,
                        collection_name=stock_news_collection_name,
                        recreate_collection=recreate_stock_news_collection,
                    )
                    st.sidebar.success(f"Ingested {len(news_documents)} articles into '{stock_news_collection_name}'.")
                else:
                    st.sidebar.warning(f"No articles found for tickers: {stock_tickers_input}")
            except Exception as e:
                st.sidebar.error(f"Error during ingestion: {e}")

# Web Scraper Feed
st.sidebar.markdown("---")
st.sidebar.subheader("Web Scraper Feed 🕸️")

scraper_urls_input = st.sidebar.text_area(
    "Enter URLs to scrape (one per line)",
    height=150,
    key="scraper_urls_text_area",
    help="Provide full URLs (e.g., https://example.com/page).",
)

scraper_goal_input = st.sidebar.text_input(
    "Optional: Goal for scraping (e.g., 'extract product reviews')",
    key="scraper_goal_text_input",
    help="This goal will be stored as metadata.",
)

scraper_collection_name = st.sidebar.text_input(
    "Collection name for scraped content",
    value="scraped_content",
    key="scraper_collection_text_input",
    help="Scraped content will be stored in this collection.",
)

recreate_scraper_collection = st.sidebar.checkbox(
    "Recreate scraped content collection if it exists",
    value=False,
    key="recreate_scraper_collection_checkbox",
)

if st.sidebar.button("Ingest Scraped Content"):
    urls_to_scrape = [url.strip() for url in scraper_urls_input.splitlines() if url.strip().lower().startswith("http")]
    if not urls_to_scrape:
        st.sidebar.warning("Please enter at least one valid URL to scrape (must start with http/https).")
    else:
        user_goal = scraper_goal_input.strip() or None
        with st.spinner(f"Scraping & ingesting {len(urls_to_scrape)} URL(s)..."):
            try:
                scraped_documents = scrape_urls_as_documents(urls=urls_to_scrape, user_goal_for_scraping=user_goal)
                if scraped_documents:
                    engine.ingest(
                        direct_documents=scraped_documents,
                        collection_name=scraper_collection_name,
                        recreate_collection=recreate_scraper_collection,
                    )
                    st.sidebar.success(
                        f"Ingested content from {len(scraped_documents)} URL(s) " f"into '{scraper_collection_name}'."
                    )
                else:
                    st.sidebar.warning(f"No content scraped from provided URLs: {urls_to_scrape}")
            except Exception as e:
                st.sidebar.error(f"Error during scraping or ingestion: {e}")

st.header("Query the Collection")

# show past messages
for msg in st.session_state.qa_chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# current question input
question = st.text_input("Your question here:", key="qa_question_input_box")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        current_q = question.strip()
        with st.spinner("Generating adaptive answer..."):
            try:
                resp = engine.run_full_rag_workflow_sync(
                    question=current_q,
                    collection_name=collection_name,
                    chat_history=st.session_state.qa_chat_history,
                )
                ai_ans = resp.get("answer", "Sorry, no answer.")

                st.session_state.qa_chat_history.append(HumanMessage(content=current_q))
                st.session_state.qa_chat_history.append(AIMessage(content=ai_ans))

            except Exception as e:
                st.session_state.qa_chat_history.append(HumanMessage(content=current_q))
                st.session_state.qa_chat_history.append(AIMessage(content=f"Error processing request: {e}"))
                st.error(f"Error: {e}")

        st.rerun()

# Insight Agent (Advanced Tasks)
st.markdown("---")
st.header("🔬 Insight Agent (Advanced Tasks)")

agent_goal_input = st.text_area(
    "Describe your complex task or high-level goal for the Insight Agent:",
    height=150,
    key="agent_goal_text_area",
    help=(
        "Example: 'Fetch the latest news for MSFT, ingest it into a new collection named "
        "'msft_daily_news', then summarize the top 3 positive developments.'"
    ),
)

if st.button("Execute Agent Task"):
    if not agent_goal_input.strip():
        st.warning("Please describe a task or goal for the Insight Agent.")
    else:
        if not engine:
            st.error("CoreRAGEngine is not loaded. Cannot run agent.")
            st.stop()
        agent_api_key = os.getenv("OPENAI_API_KEY")
        if not agent_api_key:
            st.error("OPENAI_API_KEY is required for the Insight Agent. Set it in your .env.")
            st.stop()

        with st.spinner("Insight Agent is planning and executing..."):
            try:
                insight_agent = AgentLoopWorkflow(
                    openai_api_key=agent_api_key,
                    model="gpt-4o",
                    core_rag_engine_instance=engine,
                    enable_tavily_search=True,
                    enable_python_repl=False,
                )
                final_state: AgentLoopState = insight_agent.run_workflow(goal=agent_goal_input.strip())

                st.subheader("Agent Execution Log & Results")
                steps = final_state.get("past_steps", [])
                if steps:
                    with st.expander("Show Agent's Steps", expanded=False):
                        for i, (action, obs) in enumerate(steps, 1):
                            st.markdown(f"**Step {i}:**")
                            if hasattr(action, "tool") and hasattr(action, "tool_input"):
                                st.markdown(f"Tool: `{action.tool}`")
                                st.code(
                                    json.dumps(action.tool_input, indent=2),
                                    language="json",
                                )
                            st.markdown("**Output:**")
                            st.write(obs)
                            st.markdown("---")
                else:
                    st.info("No intermediate tool calls were logged.")

                st.markdown("---")
                st.subheader("Agent's Final Output")
                outcome = final_state.get("agent_outcome")
                if isinstance(outcome, AgentFinish):
                    st.success("Agent Task Completed!")
                    st.markdown(outcome.return_values.get("output", ""))
                elif isinstance(outcome, AgentAction):
                    st.warning("Agent stopped at an action (expected to finish).")
                    st.code(str(outcome), language="json")
                else:
                    st.info("Agent finished without a clear outcome.")

            except Exception as e:
                st.error(f"Error running Insight Agent: {e}")
                logging.exception("Insight Agent execution error")
