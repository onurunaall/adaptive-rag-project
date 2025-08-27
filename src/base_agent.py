"""
Base Agent Architecture for Multi-Agent Insight System (MAIS)
A2A Protocol compliant foundation for all specialized agents
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

# A2A SDK imports (verified from official documentation)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard, 
    AgentSkill, 
    AgentCapabilities,
    MessageSendParams,
    SendMessageRequest
)
from a2a.utils import new_agent_text_message

# MCP SDK imports (verified from official documentation)
from mcp.server.fastmcp import FastMCP
from mcp.server.session import ServerSession
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

class AgentRole(str, Enum):
    """Specialized agent roles in the system"""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher" 
    ANALYST = "analyst"
    WRITER = "writer"
    CODER = "coder"
    CRITIC = "critic"
    MEMORY_KEEPER = "memory_keeper"
    TOOL_MASTER = "tool_master"
    WEB_NAVIGATOR = "web_navigator"
    DATA_SCIENTIST = "data_scientist"

class AgentStatus(str, Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class AgentCapability:
    """Enhanced capability definition for agents"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    requires_auth: bool = False
    estimated_duration: Optional[int] = None  # seconds
    resource_intensive: bool = False
    dependencies: List[str] = field(default_factory=list)

@dataclass 
class AgentContext:
    """Extended context for agent operations"""
    task_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 1  # 1=low, 5=critical
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None
    collaboration_context: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """
    Base class for all A2A compliant agents in the system.
    Provides core functionality for agent communication, MCP tool access,
    and collaboration with other agents.
    """
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        description: str,
        capabilities: List[AgentCapability],
        llm_config: Optional[Dict[str, Any]] = None,
        mcp_servers: Optional[List[str]] = None,
        port: int = 0  # 0 = auto-assign
    ):
        self.name = name
        self.role = role
        self.description = description
        self.capabilities = capabilities
        self.agent_id = str(uuid.uuid4())
        self.llm_config = llm_config or {}
        self.mcp_servers = mcp_servers or []
        self.port = port
        self.status = AgentStatus.INITIALIZING
        
        # Core components
        self.logger = self._setup_logger()
        self.task_store = InMemoryTaskStore()
        self.mcp_connections: Dict[str, ClientSession] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # A2A Server components
        self.agent_executor = None
        self.request_handler = None
        self.server_app = None
        self.agent_card = None
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0.0,
            "total_processing_time": 0.0,
            "collaborations": 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated logger for this agent"""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.agent_id[:8]}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """
        Initialize the agent - setup A2A server, connect to MCP servers, etc.
        This must be called before the agent can start processing tasks.
        """
        try:
            self.logger.info(f"Initializing agent {self.name} ({self.role.value})")
            
            # Create agent executor
            self.agent_executor = self._create_agent_executor()
            
            # Setup MCP connections
            await self._setup_mcp_connections()
            
            # Create agent card
            self.agent_card = await self._create_agent_card()
            
            # Setup A2A server
            self._setup_a2a_server()
            
            self.status = AgentStatus.READY
            self.logger.info(f"Agent {self.name} initialized successfully on port {self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    def _create_agent_executor(self) -> 'AgentExecutorImpl':
        """Create the A2A agent executor for this agent"""
        return AgentExecutorImpl(self)
    
    async def _setup_mcp_connections(self):
        """Setup connections to configured MCP servers"""
        for server_path in self.mcp_servers:
            try:
                # Determine if it's a Python or Node.js server
                is_python = server_path.endswith('.py')
                command = "python" if is_python else "node"
                
                server_params = StdioServerParameters(
                    command=command,
                    args=[server_path],
                    env=None
                )
                
                # Create connection
                stdio_transport = await stdio_client(server_params)
                session = ClientSession(stdio_transport[0], stdio_transport[1])
                await session.initialize()
                
                self.mcp_connections[server_path] = session
                self.logger.info(f"Connected to MCP server: {server_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to connect to MCP server {server_path}: {e}")
    
    async def _create_agent_card(self) -> AgentCard:
        """Create A2A agent card with capabilities and skills"""
        # Convert capabilities to A2A format
        a2a_capabilities = []
        for cap in self.capabilities:
            a2a_cap = AgentCapabilities(
                name=cap.name,
                description=cap.description,
                input_schema=cap.input_schema,
                output_schema=cap.output_schema
            )
            a2a_capabilities.append(a2a_cap)
        
        # Create skills based on role
        skills = await self._define_agent_skills()
        
        return AgentCard(
            name=self.name,
            version="1.0.0",
            description=self.description,
            capabilities=a2a_capabilities,
            skills=skills,
            metadata={
                "role": self.role.value,
                "agent_id": self.agent_id,
                "status": self.status.value,
                "mcp_servers": self.mcp_servers,
                "collaboration_enabled": True
            }
        )
    
    async def _define_agent_skills(self) -> List[AgentSkill]:
        """Define agent skills based on role and capabilities"""
        base_skills = [
            AgentSkill(
                name="collaborate",
                description="Collaborate with other agents on complex tasks",
                capabilities=["message_exchange", "task_delegation"],
                examples=["Work with researcher agent to gather information"]
            ),
            AgentSkill(
                name="status_report",
                description="Provide status updates on current tasks",
                capabilities=["status_monitoring"],
                examples=["Report task progress", "Health check response"]
            )
        ]
        
        # Add role-specific skills
        role_skills = await self._get_role_specific_skills()
        return base_skills + role_skills
    
    @abstractmethod
    async def _get_role_specific_skills(self) -> List[AgentSkill]:
        """Get skills specific to this agent's role - must be implemented by subclasses"""
        pass
    
    def _setup_a2a_server(self):
        """Setup the A2A server for this agent"""
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.agent_executor,
            task_store=self.task_store
        )
        
        server_builder = A2AStarletteApplication(
            agent_card=self.agent_card,
            http_handler=self.request_handler
        )
        
        self.server_app = server_builder.build()
    
    async def start_server(self, host: str = "localhost", port: Optional[int] = None):
        """Start the A2A server for this agent"""
        import uvicorn
        
        actual_port = port or self.port or 8000
        self.port = actual_port
        
        config = uvicorn.Config(
            app=self.server_app,
            host=host,
            port=actual_port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    # Abstract methods that must be implemented by specialized agents
    @abstractmethod
    async def process_task(self, task_context: AgentContext, message: str) -> str:
        """Process a task - core agent logic"""
        pass
    
    @abstractmethod
    async def can_handle_task(self, task_description: str) -> bool:
        """Determine if this agent can handle a given task"""
        pass
    
    # Collaboration methods
    async def collaborate_with_agent(
        self, 
        target_agent_url: str, 
        task_description: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Collaborate with another agent"""
        try:
            from a2a.client.client import A2AClient
            import httpx
            
            async with httpx.AsyncClient() as client:
                a2a_client = await A2AClient.get_client_from_agent_card_url(
                    client, target_agent_url
                )
                
                request = SendMessageRequest(
                    params=MessageSendParams(
                        message={
                            'role': 'user',
                            'parts': [{'type': 'text', 'text': task_description}],
                            'messageId': str(uuid.uuid4())
                        }
                    )
                )
                
                response = await a2a_client.send_message(request)
                
                # Record collaboration
                self.collaboration_history.append({
                    "timestamp": datetime.now(),
                    "target_agent": target_agent_url,
                    "task": task_description,
                    "response": response.dict() if response else None
                })
                self.metrics["collaborations"] += 1
                
                return response.result if response else "No response received"
                
        except Exception as e:
            self.logger.error(f"Collaboration failed: {e}")
            return f"Collaboration error: {str(e)}"
    
    # MCP integration methods
    async def use_mcp_tool(
        self, 
        server_path: str, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """Use an MCP tool from a connected server"""
        try:
            if server_path not in self.mcp_connections:
                raise ValueError(f"Not connected to MCP server: {server_path}")
            
            session = self.mcp_connections[server_path]
            result = await session.call_tool(tool_name, arguments)
            
            self.logger.debug(f"MCP tool {tool_name} result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"MCP tool call failed: {e}")
            raise
    
    async def get_mcp_resources(self, server_path: str) -> List[Dict[str, Any]]:
        """Get available resources from an MCP server"""
        try:
            if server_path not in self.mcp_connections:
                raise ValueError(f"Not connected to MCP server: {server_path}")
            
            session = self.mcp_connections[server_path]
            response = await session.list_resources()
            return [resource.dict() for resource in response.resources]
            
        except Exception as e:
            self.logger.error(f"Failed to get MCP resources: {e}")
            return []
    
    # Utility methods
    def update_status(self, status: AgentStatus, message: Optional[str] = None):
        """Update agent status"""
        old_status = self.status
        self.status = status
        self.logger.info(f"Status changed from {old_status} to {status}")
        if message:
            self.logger.info(f"Status message: {message}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            **self.metrics,
            "status": self.status.value,
            "uptime": (datetime.now() - datetime.now()).total_seconds(),  # TODO: track actual uptime
            "active_tasks": len(self.task_store._tasks) if hasattr(self.task_store, '_tasks') else 0
        }
    
    async def shutdown(self):
        """Clean shutdown of the agent"""
        self.logger.info(f"Shutting down agent {self.name}")
        
        # Close MCP connections
        for session in self.mcp_connections.values():
            try:
                await session.close()
            except Exception as e:
                self.logger.error(f"Error closing MCP connection: {e}")
        
        self.status = AgentStatus.OFFLINE
        self.logger.info("Agent shutdown complete")


class AgentExecutorImpl(AgentExecutor):
    """
    A2A Agent Executor implementation that bridges A2A protocol 
    with our BaseAgent functionality
    """
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute agent task following A2A protocol"""
        try:
            self.agent.update_status(AgentStatus.BUSY)
            start_time = datetime.now()
            
            # Extract message from context
            message_content = self._extract_message_content(context)
            
            # Create agent context
            agent_context = AgentContext(
                task_id=str(uuid.uuid4()),
                metadata={"a2a_context": context.dict() if hasattr(context, 'dict') else str(context)}
            )
            
            # Process the task
            result = await self.agent.process_task(agent_context, message_content)
            
            # Send result back through event queue
            event_queue.enqueue_event(new_agent_text_message(result))
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, success=True)
            
            self.agent.update_status(AgentStatus.READY)
            
        except Exception as e:
            self.agent.logger.error(f"Task execution failed: {e}")
            error_message = f"Error processing task: {str(e)}"
            event_queue.enqueue_event(new_agent_text_message(error_message))
            
            self._update_metrics(0, success=False)
            self.agent.update_status(AgentStatus.ERROR)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel ongoing task"""
        self.agent.logger.info("Task cancellation requested")
        self.agent.update_status(AgentStatus.READY)
        event_queue.enqueue_event(new_agent_text_message("Task cancelled"))
    
    def _extract_message_content(self, context: RequestContext) -> str:
        """Extract message content from A2A request context"""
        # This is simplified - in practice you'd parse the A2A message format
        return str(context)  # TODO: Implement proper message parsing
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update agent performance metrics"""
        if success:
            self.agent.metrics["tasks_completed"] += 1
        else:
            self.agent.metrics["tasks_failed"] += 1
        
        self.agent.metrics["total_processing_time"] += processing_time
        
        # Update average response time
        total_tasks = self.agent.metrics["tasks_completed"] + self.agent.metrics["tasks_failed"]
        if total_tasks > 0:
            self.agent.metrics["average_response_time"] = (
                self.agent.metrics["total_processing_time"] / total_tasks
            )


# Example specialized agent implementation
class ExampleSpecializedAgent(BaseAgent):
    """
    Example implementation showing how to create a specialized agent
    """
    
    async def _get_role_specific_skills(self) -> List[AgentSkill]:
        """Define skills specific to this example agent"""
        return [
            AgentSkill(
                name="example_task",
                description="Perform example specialized tasks",
                capabilities=["text_processing"],
                examples=["Process and analyze text input"]
            )
        ]
    
    async def process_task(self, task_context: AgentContext, message: str) -> str:
        """Process task - example implementation"""
        self.logger.info(f"Processing task: {message}")
        return f"Processed by {self.name}: {message}"
    
    async def can_handle_task(self, task_description: str) -> bool:
        """Determine if this agent can handle the task"""
        # Simple keyword-based matching for example
        keywords = ["example", "test", "demo"]
        return any(keyword in task_description.lower() for keyword in keywords)


# Usage example
async def create_example_agent() -> BaseAgent:
    """Create an example agent for testing"""
    capabilities = [
        AgentCapability(
            name="text_processing",
            description="Process and analyze text input",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"result": {"type": "string"}}}
        )
    ]
    
    agent = ExampleSpecializedAgent(
        name="Example Agent",
        role=AgentRole.RESEARCHER,
        description="Example agent for testing and demonstration",
        capabilities=capabilities,
        port=8001
    )
    
    await agent.initialize()
    return agent
