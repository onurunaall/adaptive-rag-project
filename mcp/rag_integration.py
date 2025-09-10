try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MultiServerMCPClient = None
    MCP_AVAILABLE = False

from langchain.tools import BaseTool, Tool
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import logging
from pathlib import Path
from langchain.schema import Document
import numpy as np
from src.config import settings as app_settings

PROJECT_ROOT = Path(__file__).parent.parent

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
        
        # Check if MCP is available
        if not MCP_AVAILABLE:
            self.logger.warning("MCP adapters not available. MCP functionality will be disabled.")
            self.mcp_available = False
            return
        
        self.mcp_available = True
        
        filesystem_args: list[str] = [str(PROJECT_ROOT / "mcp" / "filesystem_server.py")]
        memory_args: list[str] = [str(PROJECT_ROOT / "mcp" / "memory_server.py")]
        sql_args: list[str] = [str(PROJECT_ROOT / "mcp" / "sql_server.py")]

        # Build MCP configuration based on enabled features
        self.mcp_config = mcp_config or {}
        
        if enable_filesystem:
            self.mcp_config["filesystem"] = {
                "command": getattr(app_settings.mcp, 'filesystem_command', 'python'),
                "args": getattr(app_settings.mcp, 'filesystem_args', filesystem_args),
                "transport": getattr(app_settings.mcp, 'filesystem_transport', "stdio")
            }
        
        if enable_memory:
            self.mcp_config["memory"] = {
                "command": getattr(app_settings.mcp, 'memory_command', 'python'),
                "args": getattr(app_settings.mcp, 'memory_args', memory_args),
                "transport": getattr(app_settings.mcp, 'memory_transport', "stdio")
            }
        
        if enable_sql:
            self.mcp_config["sql"] = {
                "command": getattr(app_settings.mcp, 'sql_command', 'python'),
                "args": getattr(app_settings.mcp, 'sql_args', sql_args),
                "transport": getattr(app_settings.mcp, 'sql_transport', "stdio")
            }
        
        # State for monitoring
        self.monitoring_active = False
        self.monitoring_tasks = {}
        self.deleted_files_handler = None
    
    async def initialize_mcp_servers(self) -> bool:
        """Initialize and connect to MCP servers with error handling."""
        if not self.mcp_available:
            self.logger.warning("MCP not available - skipping server initialization")
            return False
            
        try:
            if not self.mcp_config:
                self.logger.warning("No MCP servers configured")
                return False
            
            self.logger.info(f"Initializing {len(self.mcp_config)} MCP servers...")
            
            self.mcp_client = MultiServerMCPClient(self.mcp_config)
            
            # Get tools from all servers
            tools = await asyncio.wait_for(self.mcp_client.get_tools(), timeout=30.0)
            
            # Store tools by name for easy access
            for tool in tools:
                self.tools[tool.name] = tool
            
            self.logger.info(f"Successfully loaded {len(self.tools)} MCP tools")
            return True
            
        except asyncio.TimeoutError:
            self.logger.error("MCP server initialization timed out")
            return False
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
        use_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced RAG with conversation memory integration and semantic search.

        Args:
            question: User's question
            session_id: Session identifier
            collection_name: RAG collection to use
            use_memory_context: Whether to include past conversations
            store_conversation: Whether to store this conversation
            use_embeddings: Whether to use embeddings for semantic search

        Returns:
            RAG response with memory context
        """
        enhanced_context = ""
        memory_results = []
        query_embedding = None

        # Generate query embedding if needed
        if use_embeddings and use_memory_context:
            try:
                query_embedding = self.rag.embedding_model.embed_query(question)
            except Exception as e:
                self.logger.error(f"Failed to generate query embedding: {e}")
                use_embeddings = False

        # Retrieve relevant memories if requested
        if use_memory_context and self.tools.get("retrieve_relevant_memories"):
            try:
                memories = await self.tools["retrieve_relevant_memories"].ainvoke({
                    "query": question,
                    "session_id": session_id,
                    "top_k": 3,
                    "similarity_threshold": 0.3,
                    "query_embedding": query_embedding if use_embeddings else None
                })

                if memories:
                    memory_context_parts = []
                    for mem in memories:
                        if not isinstance(mem, dict) or "error" in mem:
                            continue
                        memory_context_parts.append(
                            f"Previous Q: {mem.get('question', '')}\n"
                            f"Previous A: {mem.get('answer', '')}"
                        )

                    if memory_context_parts:
                        enhanced_context = "Relevant conversation history:\n" + \
                                           "\n---\n".join(memory_context_parts) + \
                                           "\n\nCurrent question: "

                    memory_results = memories

            except Exception as e:
                self.logger.error(f"Error retrieving memories: {e}")

        # Run RAG with enhanced question
        full_question = enhanced_context + question if enhanced_context else question

        result = self.rag.run_full_rag_workflow(
            question=full_question,
            collection_name=collection_name
        )

        # Store conversation if requested
        if store_conversation and self.tools.get("store_conversation_context"):
            try:
                # Generate embedding for the new conversation
                conversation_embedding = None
                if use_embeddings:
                    try:
                        conv_text = f"Q: {question}\nA: {result.get('answer', '')}"
                        conversation_embedding = self.rag.embedding_model.embed_query(conv_text)
                    except Exception as e:
                        self.logger.error(f"Failed to generate conversation embedding: {e}")

                await self.tools["store_conversation_context"].ainvoke({
                    "session_id": session_id,
                    "question": question,
                    "answer": result.get("answer", ""),
                    "context_docs": [
                        {"source": s.get("source"), "preview": s.get("preview")}
                        for s in result.get("sources", [])
                    ],
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "collection": collection_name or "default"
                    },
                    "embedding_vector": conversation_embedding
                })

            except Exception as e:
                self.logger.error(f"Error storing conversation: {e}")

        # Add memory context to result
        result["memory_context_used"] = len(memory_results) > 0
        result["memories_retrieved"] = len(memory_results)

        return result
