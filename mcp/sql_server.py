from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import BaseTool, Tool
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import logging
from pathlib import Path
from langchain.schema import Document
import numpy as np
from src.config import settings as app_settings

class MCPEnhancedRAG:
    """
    Enhanced RAG system with MCP integration for:
    - Real-time document monitoring and ingestion
    - Conversation memory management with embeddings
    - SQL database integration
    - Robust error handling
    """
    
    def __init__(
        self, 
        core_rag_engine, 
        mcp_config: Dict[str, Any] = None,
        enable_filesystem: bool = True,
        enable_memory: bool = True,
        enable_sql: bool = True
    ):
        """
        Initialize MCP-enhanced RAG system.
        
        Args:
            core_rag_engine: Instance of CoreRAGEngine
            mcp_config: Configuration for MCP servers
            enable_filesystem: Whether to enable filesystem MCP
            enable_memory: Whether to enable memory MCP
            enable_sql: Whether to enable SQL MCP
        """
        self.rag = core_rag_engine
        self.mcp_client = None
        self.tools: Dict[str, BaseTool] = {}
        self.logger = logging.getLogger(__name__)
        
        # Build MCP configuration based on enabled features
        self.mcp_config = mcp_config or {}
        
        if enable_filesystem:
            self.mcp_config["filesystem"] = {
                "command": getattr(app_settings.mcp, 'filesystem_command', 'python'),
                "args": getattr(app_settings.mcp, 'filesystem_args', ["src/mcp/filesystem_server.py"]),
                "transport": getattr(app_settings.mcp, 'filesystem_transport', "stdio")
            }
        
        if enable_memory:
            self.mcp_config["memory"] = {
                "command": getattr(app_settings.mcp, 'memory_command', 'python'),
                "args": getattr(app_settings.mcp, 'memory_args', ["src/mcp/memory_server.py"]),
                "transport": getattr(app_settings.mcp, 'memory_transport', "stdio")
            }
        
        if enable_sql:
            self.mcp_config["sql"] = {
                "command": getattr(app_settings.mcp, 'sql_command', 'python'),
                "args": getattr(app_settings.mcp, 'sql_args', ["src/mcp/sql_server.py"]),
                "transport": getattr(app_settings.mcp, 'sql_transport', "stdio")
            }
        
        # State for monitoring
        self.monitoring_active = False
        self.monitoring_tasks = {}
        self.deleted_files_handler = None
    
    async def initialize_mcp_servers(self) -> bool:
        """Initialize and connect to MCP servers with error handling."""
        try:
            if not self.mcp_config:
                self.logger.warning("No MCP servers configured")
                return False
            
            self.logger.info(f"Initializing {len(self.mcp_config)} MCP servers...")
            
            self.mcp_client = MultiServerMCPClient(self.mcp_config)
            
            # Get tools from all servers
            tools = await self.mcp_client.get_tools()
            
            # Store tools by name for easy access
            for tool in tools:
                self.tools[tool.name] = tool
            
            self.logger.info(f"Successfully loaded {len(self.tools)} MCP tools")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP servers: {e}")
            return False
    
    def set_deleted_files_handler(self, handler):
        """
        Set a custom handler for deleted files.
        
        Args:
            handler: Callable that takes a list of deleted file paths
        """
        self.deleted_files_handler = handler
    
    async def auto_ingest_with_monitoring(
        self, 
        watch_dir: str,
        collection_name: str = "auto_ingested",
        file_types: List[str] = None,
        poll_interval: int = 10,
        use_watchdog: bool = True
    ):
        """
        Monitor directory and auto-ingest new or modified documents.
        Handles deleted files properly.
        
        Args:
            watch_dir: Directory to monitor
            collection_name: Collection to store documents
            file_types: File types to monitor
            poll_interval: Seconds between checks (for polling mode)
            use_watchdog: Whether to use watchdog for real-time monitoring
        """
        if not self.tools.get("monitor_directory"):
            error_msg = "Filesystem MCP server not available"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.monitoring_active = True
        file_types = file_types or ["pdf", "txt", "md", "docx"]
        
        self.logger.info(f"Starting auto-ingestion monitoring for {watch_dir}")
        
        # Try to start real-time monitoring with watchdog
        if use_watchdog and self.tools.get("start_monitoring"):
            try:
                result = await self.tools["start_monitoring"].ainvoke({
                    "path": watch_dir,
                    "file_types": file_types
                })
                
                if result.get("success"):
                    self.logger.info("Started real-time monitoring with watchdog")
                    use_watchdog = True
                else:
                    self.logger.warning("Failed to start watchdog, falling back to polling")
                    use_watchdog = False
            except Exception as e:
                self.logger.warning(f"Watchdog not available: {e}, using polling")
                use_watchdog = False
        
        # Monitoring loop
        while self.monitoring_active:
            try:
                if use_watchdog and self.tools.get("get_file_changes"):
                    # Get changes from watchdog
                    changes = await self.tools["get_file_changes"].ainvoke({})
                else:
                    # Use polling fallback
                    changes = await self.tools["monitor_directory"].ainvoke({
                        "path": watch_dir,
                        "file_types": file_types,
                        "recursive": True
                    })
                
                if changes.get("has_changes") or changes.get("new_files") or changes.get("modified_files"):
                    # Process new files
                    for file_path in changes.get("new_files", []):
                        try:
                            self.logger.info(f"Ingesting new file: {file_path}")
                            
                            # Get metadata
                            metadata = await self.tools["get_file_metadata"].ainvoke({
                                "filepath": file_path
                            })
                            
                            if "error" in metadata:
                                self.logger.error(f"Failed to get metadata for {file_path}: {metadata['error']}")
                                continue
                            
                            # Ingest based on file type
                            if file_path.endswith('.pdf'):
                                source_type = "pdf_path"
                            elif file_path.endswith(('.txt', '.md')):
                                source_type = "text_path"
                            else:
                                continue
                            
                            self.rag.ingest(
                                sources=[{"type": source_type, "value": file_path}],
                                collection_name=collection_name,
                                recreate_collection=False
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Failed to ingest {file_path}: {e}")
                    
                    # Process modified files
                    for file_path in changes.get("modified_files", []):
                        try:
                            self.logger.info(f"Re-ingesting modified file: {file_path}")
                            
                            if file_path.endswith('.pdf'):
                                source_type = "pdf_path"
                            elif file_path.endswith(('.txt', '.md')):
                                source_type = "text_path"
                            else:
                                continue
                            
                            # TODO: Remove old version from vector store first
                            self.rag.ingest(
                                sources=[{"type": source_type, "value": file_path}],
                                collection_name=collection_name,
                                recreate_collection=False
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Failed to re-ingest {file_path}: {e}")
                    
                    # Handle deleted files
                    deleted_files = changes.get("deleted_files", [])
                    if deleted_files:
                        self.logger.info(f"Detected {len(deleted_files)} deleted files")
                        
                        if self.deleted_files_handler:
                            try:
                                await self.deleted_files_handler(deleted_files, collection_name)
                            except Exception as e:
                                self.logger.error(f"Failed to handle deleted files: {e}")
                        else:
                            self.logger.warning("No handler configured for deleted files")
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(poll_interval)
    
    async def rag_with_memory(
        self,
        question: str,
        session_id: str,
        collection_name: Optional[str] = None,
        use_memory_context: bool = True,
        store_conversation: bool = True,
        use_embeddings:
