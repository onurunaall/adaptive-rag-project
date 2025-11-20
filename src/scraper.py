from typing import List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
import logging
from urllib.parse import urlparse
import ipaddress


def _validate_url_security(url: str) -> tuple[bool, str]:
    """
    Validate URL to prevent SSRF attacks.

    Args:
        url: URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)

        # Only allow http and https schemes
        if parsed.scheme not in ('http', 'https'):
            return False, f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed."

        # Block localhost and loopback addresses
        if parsed.hostname:
            hostname_lower = parsed.hostname.lower()
            if hostname_lower in ('localhost', '127.0.0.1', '::1', '0.0.0.0'):
                return False, f"Access to localhost/loopback addresses is not allowed: {hostname_lower}"

            # Block private IP ranges (optional but recommended)
            try:
                ip = ipaddress.ip_address(hostname_lower)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False, f"Access to private IP addresses is not allowed: {ip}"
            except ValueError:
                # Not an IP address, likely a domain name - allow it
                pass

        return True, ""
    except Exception as e:
        return False, f"Invalid URL format: {e}"


def scrape_urls_as_documents(urls: List[str], user_goal_for_scraping: Optional[str] = None) -> List[Document]:
    """
    Scrapes content from a list of URLs using WebBaseLoader and returns them as LangChain Documents.

    Security: Only allows http/https schemes and blocks access to localhost/private IPs to prevent SSRF attacks.

    Args:
        urls: A list of URLs to scrape.
        user_goal_for_scraping: Optional. A string describing the purpose of scraping,
                               to be added to document metadata.

    Returns:
        A list of Document objects.
    """
    if not urls:
        return []

    # Validate all URLs for security
    validated_urls = []
    for url in urls:
        is_valid, error_msg = _validate_url_security(url)
        if not is_valid:
            logging.warning(f"Blocked potentially unsafe URL: {url}. Reason: {error_msg}")
            continue
        validated_urls.append(url)

    if not validated_urls:
        logging.error("All URLs were blocked by security validation")
        return []

    all_docs = []
    try:
        loader = WebBaseLoader(
            web_paths=validated_urls,
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
