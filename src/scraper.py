from typing import List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
import logging


def scrape_urls_as_documents(urls: List[str], user_goal_for_scraping: Optional[str] = None) -> List[Document]:
    """
    Scrapes content from a list of URLs using WebBaseLoader and returns them as LangChain Documents.

    Args:
        urls: A list of URLs to scrape.
        user_goal_for_scraping: Optional. A string describing the purpose of scraping,
                               to be added to document metadata.

    Returns:
        A list of Document objects.
    """
    if not urls:
        return []

    all_docs = []
    try:
        loader = WebBaseLoader(
            web_paths=urls,
            # Optional: Configure WebBaseLoader further if needed, e.g., with bs_get_text_kwargs
            # For example, to try and get cleaner text:
            # bs_get_text_kwargs={"separator": " ", "strip": True}
        )
        loaded_docs = loader.load()  # returns a list of Document objects

        for doc in loaded_docs:
            if user_goal_for_scraping:
                doc.metadata["user_goal_for_scraping"] = user_goal_for_scraping
            if "source" not in doc.metadata and hasattr(doc, "metadata") and "url" in doc.metadata:
                doc.metadata["source"] = doc.metadata["url"]
            all_docs.append(doc)

    except Exception as e:
        logging.error(f"Error scraping URLs {urls}: {e}", exc_info=True)
        return []

    return all_docs
