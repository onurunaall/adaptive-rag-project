"""
stock.py: Stock analysis agent workflow using LangGraph.

Fetches stock news from Yahoo Finance, analyzes it, and suggests trading ideas.
LangGraph coordinates agents; Streamlit provides the interface.
"""

from typing import List, Dict, Union
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.schema import Document
import logging

def fetch_stock_news_documents(tickers_input: Union[str, List[str]], max_articles_per_ticker: int) -> List[Document]:
    """
    Fetches recent stock news articles for given tickers using YahooFinanceNewsTool
    and returns them as a list of LangChain Document objects.

    Args:
        tickers_input: A single ticker string, a comma-separated string of tickers, or a list of ticker strings.
        max_articles_per_ticker: The maximum number of articles to fetch per ticker. 
                                 Note: YahooFinanceNewsTool might have its own internal limits or behavior regarding this.
                                 The tool itself takes a single string of comma-separated tickers.

    Returns:
        A list of Document objects, where each document represents a news article.
    """
    if isinstance(tickers_input, list):
        tickers_str = ",".join(tickers_input)
    else:
        tickers_str = tickers_input

    if not tickers_str:
        return []

    tool = YahooFinanceNewsTool(top_n=max_articles_per_ticker)

    try:
        # The tool expects a single string of comma-separated tickers.
        # It returns a list of dictionaries, or a string indicating no articles.
        results = tool.run(tickers_str)
    except Exception as e:
        print(f"Error fetching news for {tickers_str}: {e}")
        return []

    documents = []
    if isinstance(results, str) and "Cannot find any article" in results:
        print(f"No articles found for tickers: {tickers_str}")
        return []

    if isinstance(results, list):
        for article_info in results:
            content = article_info.get('summary', '')
            if not content and article_info.get('title'):
                content = article_info.get('title')

            metadata = {
                "source": article_info.get('link', 'Unknown Yahoo Finance URL'),
                "title": article_info.get('title', 'No Title'),
                "published_date": article_info.get('published', 'Unknown Publish Date'),
                "tickers": tickers_str
            }
            documents.append(Document(page_content=content, metadata=metadata))

    return documents
