"""
Error handling utilities for RAG workflow state management.
"""
import logging
from typing import Any, Dict, Optional

from src.rag.models import CoreGraphState


class ErrorHandler:
    """Utility class for managing errors in RAG workflow state."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.

        Args:
            logger: Logger instance for logging
        """
        self.logger = logger or logging.getLogger(__name__)

    def append_error(self, state: CoreGraphState, error_msg: str) -> None:
        """
        Safely append an error message to the state's error_message field.

        Args:
            state: Current workflow state
            error_msg: Error message to append
        """
        try:
            current_error = state.get("error_message")

            if current_error:
                # Append with separator
                state["error_message"] = f"{current_error} | {error_msg}"
            else:
                # First error
                state["error_message"] = error_msg

            self.logger.debug(f"Error appended to state: {error_msg}")

        except Exception as e:
            # Even error handling can fail!
            self.logger.critical(f"Failed to append error to state: {e}", exc_info=True)
            # Last resort: overwrite
            state["error_message"] = f"ERROR HANDLING FAILED: {error_msg}"

    def clear_error(self, state: CoreGraphState) -> None:
        """
        Clear the error message from state.

        Args:
            state: Current workflow state
        """
        state["error_message"] = None

    def has_error(self, state: CoreGraphState) -> bool:
        """
        Check if state has an error.

        Args:
            state: Current workflow state

        Returns:
            True if state contains an error message
        """
        return bool(state.get("error_message"))

    def get_error_summary(self, state: CoreGraphState) -> Optional[Dict[str, Any]]:
        """
        Get a structured summary of errors in the state.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with error information, or None if no errors
        """
        if not self.has_error(state):
            return None

        error_msg = state.get("error_message", "")

        # Parse errors (split by delimiter)
        error_list = [e.strip() for e in error_msg.split("|") if e.strip()]

        return {
            "has_error": True,
            "error_count": len(error_list),
            "errors": error_list,
            "full_message": error_msg,
            "severity": "critical" if "critical" in error_msg.lower() else "error",
        }
