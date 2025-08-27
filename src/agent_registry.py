"""
Agent Registry & Discovery System for Multi-Agent Insight System (MAIS)
Central coordination hub for agent management, discovery, and communication
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque

import aiohttp
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import from our base agent
from .base_agent import BaseAgent, AgentRole, AgentStatus, AgentCapability, AgentContext

class EventType(str, Enum):
    """System event types for agent coordination"""
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    SYSTEM_ALERT = "system_alert"
    HEALTH_CHECK = "health_check"

class TaskPriority(int, Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 3
    HIGH = 5
    URGENT = 7
    CRITICAL = 9

@dataclass
class SystemEvent:
    """System-wide events for agent coordination"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    source_agent_id: str
    target_agent_ids: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: TaskPriority = TaskPriority.NORMAL
    processed: bool = False

@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    name: str
    role: AgentRole
    description: str
    capabilities: List[AgentCapability]
    endpoint: str
    status: AgentStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    health_score: float = 1.0  # 0.0 to 1.0
    load_factor: float = 0.0   # Current load 0.0 to 1.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskRequest:
    """Task request for agent processing"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    requirements: List[str] = field(default_factory=list)  # Required capabilities
    priority: TaskPriority = TaskPriority.NORMAL
    requester_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # seconds
    created_at: datetime = field(default_factory=datetime.now)
    assigned_agent_id: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None

class AgentMatcher:
    """Intelligent agent matching based on capabilities, load, and performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def find_best_agent(
        self,
        task: TaskRequest,
        available_agents: List[AgentRegistration]
    ) -> Optional[AgentRegistration]:
        """Find the best agent for a given task"""
        
        # Filter agents that can handle the task
        capable_agents = self._filter_capable_agents(task, available_agents)
        
        if not capable_agents:
            return None
        
        # Score and rank agents
        scored_agents = []
        for agent in capable_agents:
            score = self._calculate_agent_score(task, agent)
            scored_agents.append((score, agent))
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        
        best_agent = scored_agents[0][1]
        self.logger.info(f"Selected agent {best_agent.name} for task {task.task_id}")
        
        return best_agent
    
    def _filter_capable_agents(
        self,
        task: TaskRequest,
        agents: List[AgentRegistration]
    ) -> List[AgentRegistration]:
        """Filter agents that have required capabilities"""
        capable_agents = []
        
        for agent in agents:
            if agent.status not in [AgentStatus.READY, AgentStatus.BUSY]:
                continue
            
            # Check if agent has required capabilities
            agent_capabilities = {cap.name for cap in agent.capabilities}
            required_capabilities = set(task.requirements)
            
            if required_capabilities.issubset(agent_capabilities):
                capable_agents.append(agent)
        
        return capable_agents
    
    def _calculate_agent_score(
        self,
        task: TaskRequest,
        agent: AgentRegistration
    ) -> float:
        """Calculate a score for how well an agent fits a task"""
        score = 0.0
        
        # Base score for capability match
        score += 50.0
        
        # Health score (0-30 points)
        score += agent.health_score * 30.0
        
        # Load factor (0-20 points, lower load = higher score)
        score += (1.0 - agent.load_factor) * 20.0
        
        # Performance metrics
        if agent.performance_metrics:
            avg_response_time = agent.performance_metrics.get("average_response_time", 5.0)
            success_rate = agent.performance_metrics.get("success_rate", 0.8)
            
            # Response time score (0-10 points, faster = better)
            if avg_response_time > 0:
                response_score = max(0, 10.0 - avg_response_time)
                score += response_score
            
            # Success rate score (0-15 points)
            score += success_rate * 15.0
        
        # Priority boost for urgent tasks
        if task.priority >= TaskPriority.HIGH and agent.load_factor < 0.5:
            score += 10.0
        
        # Role-specific bonuses
        role_bonus = self._get_role_bonus(task, agent.role)
        score += role_bonus
        
        return score
    
    def _get_role_bonus(self, task: TaskRequest, agent_role: AgentRole) -> float:
        """Get bonus points for role-task alignment"""
        task_desc = task.description.lower()
        
        role_keywords = {
            AgentRole.RESEARCHER: ["research", "find", "gather", "information", "search"],
            AgentRole.ANALYST: ["analyze", "analysis", "insights", "patterns", "statistics"],
            AgentRole.WRITER: ["write", "create", "document", "report", "summary"],
            AgentRole.CODER: ["code", "program", "develop", "script", "api"],
            AgentRole.CRITIC: ["review", "evaluate", "assess", "validate", "check"],
            AgentRole.WEB_NAVIGATOR: ["web", "scrape", "browse", "url", "website"]
        }
        
        if agent_role in role_keywords:
            keywords = role_keywords[agent_role]
            matches = sum(1 for keyword in keywords if keyword in task_desc)
            return matches * 5.0  # 5 points per keyword match
        
        return 0.0

class EventBus:
    """High-performance event bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, Set[Callable]] = defaultdict(set)
        self.agent_subscribers: Dict[str, Set[EventType]] = defaultdict(set)
        self.event_history: deque = deque(maxlen=1000)  # Keep last 1000 events
        self.websocket_connections: Set[WebSocket] = set()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.events_published = 0
        self.events_processed = 0
    
    def subscribe(self, event_type: EventType, handler: Callable, agent_id: str = None):
        """Subscribe to events"""
        self.subscribers[event_type].add(handler)
        
        if agent_id:
            self.agent_subscribers[agent_id].add(event_type)
        
        self.logger.debug(f"New subscriber for {event_type}: {handler}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable, agent_id: str = None):
        """Unsubscribe from events"""
        self.subscribers[event_type].discard(handler)
        
        if agent_id:
            self.agent_subscribers[agent_id].discard(event_type)
    
    async def publish(self, event: SystemEvent):
        """Publish event to all subscribers"""
        try:
            self.events_published += 1
            self.event_history.append(event)
            
            # Call registered handlers
            handlers = self.subscribers.get(event.event_type, set())
            
            tasks = []
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler failed: {e}")
            
            # Execute async handlers
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Send to WebSocket connections
            await self._broadcast_to_websockets(event)
            
            self.events_processed += 1
            
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
    
    async def _broadcast_to_websockets(self, event: SystemEvent):
        """Broadcast event to WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "system_event",
            "event": asdict(event)
        }
        
        disconnected = set()
        for websocket in self.websocket_connections.copy():
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
        
        # Clean up disconnected WebSockets
        self.websocket_connections -= disconnected
    
    async def add_websocket(self, websocket: WebSocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.add(websocket)
        self.logger.info(f"WebSocket connected. Total: {len(self.websocket_connections)}")
    
    async def remove_websocket(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.websocket_connections.discard(websocket)
        self.logger.info(f"WebSocket disconnected. Total: {len(self.websocket_connections)}")
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events"""
        events = list(self.event_history)[-limit:]
        return [asdict(event) for event in events]

class AgentRegistry:
    """
    Central registry for agent discovery, management, and coordination.
    This is the heart of the multi-agent system.
    """
    
    def __init__(self, health_check_interval: int = 30):
        self.agents: Dict[str, AgentRegistration] = {}
        self.event_bus = EventBus()
        self.agent_matcher = AgentMatcher()
        self.pending_tasks: Dict[str, TaskRequest] = {}
        self.health_check_interval = health_check_interval
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Start background tasks
        self._health_check_task = None
        self._cleanup_task = None
        
        # Performance metrics
        self.registry_metrics = {
            "agents_registered": 0,
            "tasks_processed": 0,
            "successful_matches": 0,
            "failed_matches": 0,
            "average_response_time": 0.0
        }
    
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        self._health_check_task = asyncio.create_task(self._periodic_health_check())
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.logger.info("Background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks"""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self.logger.info("Background tasks stopped")
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent with the registry"""
        try:
            registration = AgentRegistration(
                agent_id=agent.agent_id,
                name=agent.name,
                role=agent.role,
                description=agent.description,
                capabilities=agent.capabilities,
                endpoint=f"http://localhost:{agent.port}",
                status=agent.status,
                metadata=agent.get_metrics()
            )
            
            self.agents[agent.agent_id] = registration
            self.registry_metrics["agents_registered"] += 1
            
            # Publish registration event
            event = SystemEvent(
                event_type=EventType.AGENT_REGISTERED,
                source_agent_id=agent.agent_id,
                payload={
                    "agent_name": agent.name,
                    "role": agent.role.value,
                    "capabilities": [cap.name for cap in agent.capabilities]
                }
            )
            await self.event_bus.publish(event)
            
            self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        try:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            del self.agents[agent_id]
            
            # Publish unregistration event
            event = SystemEvent(
                event_type=EventType.AGENT_UNREGISTERED,
                source_agent_id=agent_id,
                payload={"agent_name": agent.name}
            )
            await self.event_bus.publish(event)
            
            self.logger.info(f"Unregistered agent: {agent.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus, metadata: Dict[str, Any] = None):
        """Update agent status and metadata"""
        if agent_id in self.agents:
            old_status = self.agents[agent_id].status
            self.agents[agent_id].status = status
            self.agents[agent_id].last_heartbeat = datetime.now()
            
            if metadata:
                self.agents[agent_id].metadata.update(metadata)
            
            # Publish status change event
            if old_status != status:
                event = SystemEvent(
                    event_type=EventType.AGENT_STATUS_CHANGED,
                    source_agent_id=agent_id,
                    payload={
                        "old_status": old_status.value,
                        "new_status": status.value,
                        "metadata": metadata
                    }
                )
                await self.event_bus.publish(event)
    
    async def submit_task(self, task: TaskRequest) -> Optional[str]:
        """Submit a task for processing by an appropriate agent"""
        try:
            self.registry_metrics["tasks_processed"] += 1
            
            # Find the best agent for this task
            available_agents = [
                agent for agent in self.agents.values()
                if agent.status in [AgentStatus.READY, AgentStatus.BUSY]
            ]
            
            best_agent = self.agent_matcher.find_best_agent(task, available_agents)
            
            if not best_agent:
                self.registry_metrics["failed_matches"] += 1
                self.logger.warning(f"No suitable agent found for task: {task.description}")
                return None
            
            # Assign task to agent
            task.assigned_agent_id = best_agent.agent_id
            task.status = "assigned"
            self.pending_tasks[task.task_id] = task
            
            self.registry_metrics["successful_matches"] += 1
            
            # Publish task assignment event
            event = SystemEvent(
                event_type=EventType.TASK_ASSIGNED,
                source_agent_id="registry",
                target_agent_ids=[best_agent.agent_id],
                payload={
                    "task_id": task.task_id,
                    "task_description": task.description,
                    "assigned_agent": best_agent.name
                }
            )
            await self.event_bus.publish(event)
            
            # Send task to agent (this would integrate with A2A protocol)
            await self._send_task_to_agent(best_agent, task)
            
            return task.task_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit task: {e}")
            return None
    
    async def _send_task_to_agent(self, agent: AgentRegistration, task: TaskRequest):
        """Send task to agent via A2A protocol"""
        try:
            async with aiohttp.ClientSession() as session:
                # This would use the proper A2A protocol
                url = f"{agent.endpoint}/"
                
                payload = {
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": task.description
                                }
                            ],
                            "messageId": task.task_id
                        },
                        "metadata": {
                            "task_id": task.task_id,
                            "priority": task.priority.value,
                            "context": task.context
                        }
                    }
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"Task {task.task_id} sent to agent {agent.name}")
                    else:
                        self.logger.error(f"Failed to send task to agent: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send task to agent: {e}")
    
    def get_agents(self, role: Optional[AgentRole] = None, status: Optional[AgentStatus] = None) -> List[AgentRegistration]:
        """Get agents by role and/or status"""
        agents = list(self.agents.values())
        
        if role:
            agents = [a for a in agents if a.role == role]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        return agents
    
    def get_agent_by_id(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def find_agents_by_capability(self, capability: str) -> List[AgentRegistration]:
        """Find agents with specific capability"""
        matching_agents = []
        for agent in self.agents.values():
            for cap in agent.capabilities:
                if cap.name == capability:
                    matching_agents.append(agent)
                    break
        return matching_agents
    
    async def _periodic_health_check(self):
        """Periodic health check of registered agents"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                current_time = datetime.now()
                unhealthy_agents = []
                
                for agent_id, agent in self.agents.items():
                    # Check if agent hasn't sent heartbeat recently
                    time_since_heartbeat = current_time - agent.last_heartbeat
                    
                    if time_since_heartbeat > timedelta(minutes=2):
                        # Agent might be unhealthy
                        agent.health_score *= 0.9  # Reduce health score
                        
                        if agent.health_score < 0.3:
                            unhealthy_agents.append(agent_id)
                    else:
                        # Agent is healthy, restore health score
                        agent.health_score = min(1.0, agent.health_score + 0.1)
                
                # Remove unhealthy agents
                for agent_id in unhealthy_agents:
                    self.logger.warning(f"Removing unhealthy agent: {agent_id}")
                    await self.unregister_agent(agent_id)
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old tasks and events"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                current_time = datetime.now()
                
                # Clean up old completed tasks
                old_tasks = []
                for task_id, task in self.pending_tasks.items():
                    if task.status in ["completed", "failed"]:
                        age = current_time - task.created_at
                        if age > timedelta(hours=1):
                            old_tasks.append(task_id)
                
                for task_id in old_tasks:
                    del self.pending_tasks[task_id]
                
                if old_tasks:
                    self.logger.info(f"Cleaned up {len(old_tasks)} old tasks")
                
            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agent_status_counts = {}
        for status in AgentStatus:
            count = len([a for a in self.agents.values() if a.status == status])
            agent_status_counts[status.value] = count
        
        return {
            "total_agents": len(self.agents),
            "agent_status": agent_status_counts,
            "pending_tasks": len(self.pending_tasks),
            "events_published": self.event_bus.events_published,
            "events_processed": self.event_bus.events_processed,
            "registry_metrics": self.registry_metrics,
            "uptime": "TODO: implement uptime tracking"
        }


# FastAPI app for registry management
def create_registry_app(registry: AgentRegistry) -> FastAPI:
    """Create FastAPI app for registry management"""
    app = FastAPI(title="Agent Registry", description="Multi-Agent System Registry")
    
    @app.get("/agents")
    async def list_agents():
        """List all registered agents"""
        agents = list(registry.agents.values())
        return {"agents": [asdict(agent) for agent in agents]}
    
    @app.get("/agents/{agent_id}")
    async def get_agent(agent_id: str):
        """Get specific agent info"""
        agent = registry.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"agent": asdict(agent)}
    
    @app.post("/tasks")
    async def submit_task(task_data: dict):
        """Submit a task for processing"""
        task = TaskRequest(
            description=task_data["description"],
            requirements=task_data.get("requirements", []),
            priority=TaskPriority(task_data.get("priority", TaskPriority.NORMAL)),
            context=task_data.get("context", {})
        )
        
        task_id = await registry.submit_task(task)
        if task_id:
            return {"task_id": task_id, "status": "submitted"}
        else:
            raise HTTPException(status_code=400, detail="No suitable agent found")
    
    @app.get("/system/status")
    async def get_system_status():
        """Get system status"""
        return registry.get_system_status()
    
    @app.get("/events")
    async def get_recent_events():
        """Get recent system events"""
        events = registry.event_bus.get_recent_events()
        return {"events": events}
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time system updates"""
        await websocket.accept()
        await registry.event_bus.add_websocket(websocket)
        
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await registry.event_bus.remove_websocket(websocket)
    
    return app


# Usage example
async def create_registry_system(port: int = 9000) -> Tuple[AgentRegistry, FastAPI]:
    """Create and start the registry system"""
    registry = AgentRegistry()
    await registry.start_background_tasks()
    
    app = create_registry_app(registry)
    
    return registry, app


async def run_registry_server(port: int = 9000):
    """Run the registry server"""
    registry, app = await create_registry_system(port)
    
    config = uvicorn.Config(app=app, host="localhost", port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        await registry.stop_background_tasks()


if __name__ == "__main__":
    asyncio.run(run_registry_server())
