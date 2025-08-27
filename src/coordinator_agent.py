import asyncio
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from fastmcp import FastMCP
from mcp.types import Tool, TextContent

from a2a.types import AgentSkill

from .base_agent import BaseAgent, AgentRole, AgentStatus, AgentCapability, AgentContext, TaskPriority)
from .agent_registry import AgentRegistry, TaskRequest, EventType, SystemEvent

class TaskComplexity(str, Enum):
    """Task complexity levels for planning"""
    SIMPLE = "simple"          # Single agent, single step
    MODERATE = "moderate"      # Single agent, multiple steps
    COMPLEX = "complex"        # Multiple agents, coordinated
    ENTERPRISE = "enterprise"  # Multi-phase, long-running

class SubTaskStatus(str, Enum):
    """Status of individual sub-tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class SubTask:
    """Individual sub-task in a complex workflow"""
    subtask_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    required_role: AgentRole = AgentRole.RESEARCHER
    required_capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other subtask IDs
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: int = 300  # seconds
    assigned_agent_id: Optional[str] = None
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class TaskPlan:
    """Complete execution plan for a complex task"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_task: str = ""
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    subtasks: List[SubTask] = field(default_factory=list)
    execution_strategy: str = "sequential"  # sequential, parallel, hybrid
    total_estimated_duration: int = 0
    dependencies_graph: Dict[str, List[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "planning"
    progress: float = 0.0  # 0.0 to 1.0
    final_result: Optional[Dict[str, Any]] = None

class TaskPlanner:
    """Intelligent task decomposition and planning"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Task decomposition patterns
        self.decomposition_patterns = {
            "research_and_analysis": [
                ("research", AgentRole.RESEARCHER, ["information_gathering", "web_search"]),
                ("analysis", AgentRole.ANALYST, ["data_analysis", "pattern_recognition"]),
                ("synthesis", AgentRole.WRITER, ["content_generation", "summarization"])
            ],
            "content_creation": [
                ("research", AgentRole.RESEARCHER, ["information_gathering"]),
                ("outline", AgentRole.WRITER, ["content_planning"]),
                ("draft", AgentRole.WRITER, ["content_generation"]),
                ("review", AgentRole.CRITIC, ["quality_assessment"])
            ],
            "code_development": [
                ("requirements", AgentRole.ANALYST, ["requirement_analysis"]),
                ("design", AgentRole.CODER, ["system_design"]),
                ("implementation", AgentRole.CODER, ["code_generation"]),
                ("review", AgentRole.CRITIC, ["code_review"]),
                ("testing", AgentRole.CODER, ["testing"])
            ],
            "comprehensive_report": [
                ("information_gathering", AgentRole.RESEARCHER, ["web_search", "document_retrieval"]),
                ("data_analysis", AgentRole.ANALYST, ["statistical_analysis", "pattern_recognition"]),
                ("insights_extraction", AgentRole.ANALYST, ["insight_generation"]),
                ("content_structuring", AgentRole.WRITER, ["content_planning"]),
                ("report_writing", AgentRole.WRITER, ["content_generation"]),
                ("quality_review", AgentRole.CRITIC, ["quality_assessment"]),
                ("final_polish", AgentRole.WRITER, ["editing"])
            ]
        }
    
    async def analyze_task_complexity(self, task_description: str) -> TaskComplexity:
        """Analyze task complexity using LLM"""
        
        # Simple heuristics for now (can be enhanced with LLM)
        task_lower = task_description.lower()
        
        # Keywords indicating complexity
        simple_keywords = ["define", "explain", "what is", "find"]
        moderate_keywords = ["analyze", "compare", "summarize", "research"]
        complex_keywords = ["comprehensive", "detailed analysis", "multi-step", "workflow"]
        enterprise_keywords = ["enterprise", "production", "full system", "end-to-end"]
        
        if any(keyword in task_lower for keyword in enterprise_keywords):
            return TaskComplexity.ENTERPRISE
        elif any(keyword in task_lower for keyword in complex_keywords):
            return TaskComplexity.COMPLEX
        elif any(keyword in task_lower for keyword in moderate_keywords):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    async def create_task_plan(self, task_description: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create a comprehensive task execution plan"""
        
        complexity = await self.analyze_task_complexity(task_description)
        self.logger.info(f"Task complexity assessed as: {complexity}")
        
        plan = TaskPlan(
            original_task=task_description,
            complexity=complexity
        )
        
        # Determine decomposition pattern
        pattern = self._select_decomposition_pattern(task_description)
        
        if pattern and complexity in [TaskComplexity.COMPLEX, TaskComplexity.ENTERPRISE]:
            # Create subtasks based on pattern
            subtasks = self._create_subtasks_from_pattern(pattern, task_description, context)
            plan.subtasks = subtasks
            plan.execution_strategy = self._determine_execution_strategy(subtasks)
            plan.dependencies_graph = self._build_dependencies_graph(subtasks)
            plan.total_estimated_duration = sum(st.estimated_duration for st in subtasks)
        else:
            # Simple task - single subtask
            subtasks = [SubTask(
                description=task_description,
                required_role=self._determine_primary_role(task_description),
                required_capabilities=self._extract_required_capabilities(task_description),
                estimated_duration=300
            )]
            plan.subtasks = subtasks
            plan.execution_strategy = "sequential"
            plan.total_estimated_duration = 300
        
        self.logger.info(f"Created plan with {len(plan.subtasks)} subtasks")
        return plan
    
    def _select_decomposition_pattern(self, task_description: str) -> Optional[List[Tuple]]:
        """Select appropriate decomposition pattern"""
        task_lower = task_description.lower()
        
        if any(keyword in task_lower for keyword in ["report", "comprehensive", "analysis"]):
            return self.decomposition_patterns["comprehensive_report"]
        elif any(keyword in task_lower for keyword in ["code", "program", "develop"]):
            return self.decomposition_patterns["code_development"]
        elif any(keyword in task_lower for keyword in ["write", "create", "document"]):
            return self.decomposition_patterns["content_creation"]
        elif any(keyword in task_lower for keyword in ["research", "analyze", "investigate"]):
            return self.decomposition_patterns["research_and_analysis"]
        
        return None
    
    def _create_subtasks_from_pattern(
        self, 
        pattern: List[Tuple], 
        original_task: str,
        context: Dict[str, Any] = None
    ) -> List[SubTask]:
        """Create subtasks from decomposition pattern"""
        subtasks = []
        
        for i, (phase_name, role, capabilities) in enumerate(pattern):
            subtask = SubTask(
                description=f"{phase_name.title()}: {original_task}",
                required_role=role,
                required_capabilities=capabilities,
                priority=TaskPriority.NORMAL if i < len(pattern) - 1 else TaskPriority.HIGH,
                estimated_duration=self._estimate_phase_duration(phase_name),
                context=context or {}
            )
            
            # Add dependencies (each task depends on previous ones)
            if i > 0:
                subtask.dependencies = [subtasks[i-1].subtask_id]
            
            subtasks.append(subtask)
        
        return subtasks
    
    def _determine_execution_strategy(self, subtasks: List[SubTask]) -> str:
        """Determine optimal execution strategy"""
        # Check if tasks can run in parallel
        has_dependencies = any(subtask.dependencies for subtask in subtasks)
        
        if not has_dependencies and len(subtasks) > 1:
            return "parallel"
        elif has_dependencies and len(subtasks) > 3:
            return "hybrid"  # Mix of parallel and sequential
        else:
            return "sequential"
    
    def _build_dependencies_graph(self, subtasks: List[SubTask]) -> Dict[str, List[str]]:
        """Build task dependencies graph"""
        graph = {}
        
        for subtask in subtasks:
            graph[subtask.subtask_id] = subtask.dependencies.copy()
        
        return graph
    
    def _determine_primary_role(self, task_description: str) -> AgentRole:
        """Determine primary role for simple tasks"""
        task_lower = task_description.lower()
        
        role_keywords = {
            AgentRole.RESEARCHER: ["research", "find", "gather", "search", "information"],
            AgentRole.ANALYST: ["analyze", "analysis", "insights", "patterns", "data"],
            AgentRole.WRITER: ["write", "create", "document", "report", "summary"],
            AgentRole.CODER: ["code", "program", "develop", "script", "api"],
            AgentRole.WEB_NAVIGATOR: ["web", "scrape", "browse", "website", "url"]
        }
        
        for role, keywords in role_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                return role
        
        return AgentRole.RESEARCHER  # Default
    
    def _extract_required_capabilities(self, task_description: str) -> List[str]:
        """Extract required capabilities from task description"""
        capabilities = []
        task_lower = task_description.lower()
        
        capability_keywords = {
            "information_gathering": ["research", "find", "gather", "search"],
            "data_analysis": ["analyze", "analysis", "insights", "patterns"],
            "content_generation": ["write", "create", "generate", "document"],
            "web_search": ["web", "online", "internet", "search"],
            "code_generation": ["code", "program", "script", "develop"],
            "quality_assessment": ["review", "check", "validate", "assess"]
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities if capabilities else ["general_processing"]
    
    def _estimate_phase_duration(self, phase_name: str) -> int:
        """Estimate duration for different phases"""
        duration_map = {
            "research": 600,       # 10 minutes
            "information_gathering": 600,
            "analysis": 480,       # 8 minutes
            "data_analysis": 480,
            "synthesis": 360,      # 6 minutes
            "content_structuring": 240,  # 4 minutes
            "outline": 180,        # 3 minutes
            "draft": 420,         # 7 minutes
            "report_writing": 480,
            "review": 240,        # 4 minutes
            "quality_review": 300, # 5 minutes
            "requirements": 300,   # 5 minutes
            "design": 480,        # 8 minutes
            "implementation": 900, # 15 minutes
            "testing": 360,       # 6 minutes
            "final_polish": 180   # 3 minutes
        }
        
        return duration_map.get(phase_name.lower(), 300)  # Default 5 minutes

class CoordinatorAgent(BaseAgent):
    """
    Master coordinator agent that orchestrates complex multi-agent workflows.
    This is the brain of your multi-agent system.
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        llm_config: Optional[Dict[str, Any]] = None,
        port: int = 8000
    ):
        # Define coordinator capabilities
        capabilities = [
            AgentCapability(
                name="task_orchestration",
                description="Break down complex tasks and coordinate multiple agents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "context": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object", 
                    "properties": {
                        "plan": {"type": "object"},
                        "result": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="workflow_management",
                description="Manage and monitor multi-step workflows",
                input_schema={"type": "object"},
                output_schema={"type": "object"}
            ),
            AgentCapability(
                name="agent_coordination", 
                description="Coordinate collaboration between specialized agents",
                input_schema={"type": "object"},
                output_schema={"type": "object"}
            ),
            AgentCapability(
                name="progress_monitoring",
                description="Monitor and report on task execution progress",
                input_schema={"type": "object"},
                output_schema={"type": "object"}
            )
        ]
        
        # MCP servers for coordinator tools
        mcp_servers = [
            # Add paths to your MCP servers here when created
        ]
        
        super().__init__(
            name="Coordinator Agent",
            role=AgentRole.COORDINATOR,
            description="Master orchestrator for complex multi-agent tasks and workflows",
            capabilities=capabilities,
            llm_config=llm_config,
            mcp_servers=mcp_servers,
            port=port
        )
        
        # Coordinator-specific components
        self.agent_registry = agent_registry
        self.task_planner = TaskPlanner(llm_config or {})
        self.active_plans: Dict[str, TaskPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.coordination_metrics = {
            "plans_created": 0,
            "successful_coordinations": 0,
            "failed_coordinations": 0,
            "average_completion_time": 0.0,
            "total_tasks_orchestrated": 0
        }
    
    async def _get_role_specific_skills(self) -> List[AgentSkill]:
        """Define coordinator-specific skills"""
        return [
            AgentSkill(
                name="orchestrate_complex_task",
                description="Break down and coordinate complex multi-agent tasks",
                capabilities=["task_orchestration", "workflow_management"],
                examples=[
                    "Create comprehensive research report with analysis and recommendations",
                    "Develop and review code solution with documentation",
                    "Analyze market trends and create strategic presentation"
                ]
            ),
            AgentSkill(
                name="manage_workflow",
                description="Monitor and manage ongoing multi-agent workflows",
                capabilities=["workflow_management", "progress_monitoring"],
                examples=[
                    "Track progress of research and analysis pipeline",
                    "Coordinate parallel content creation tasks",
                    "Monitor code development and review process"
                ]
            ),
            AgentSkill(
                name="coordinate_agents",
                description="Facilitate collaboration between specialized agents",
                capabilities=["agent_coordination"],
                examples=[
                    "Connect researcher output to analyst input",
                    "Route writer draft to critic for review",
                    "Coordinate data scientist and researcher collaboration"
                ]
            )
        ]
    
    async def process_task(self, task_context: AgentContext, message: str) -> str:
        """Process coordination task - main orchestration logic"""
        try:
            self.logger.info(f"Processing coordination task: {message}")
            start_time = datetime.now()
            
            # Create task plan
            plan = await self.task_planner.create_task_plan(
                message, 
                task_context.metadata
            )
            
            self.active_plans[task_context.task_id] = plan
            self.coordination_metrics["plans_created"] += 1
            
            # Execute the plan
            if plan.complexity == TaskComplexity.SIMPLE:
                result = await self._execute_simple_plan(plan)
            else:
                result = await self._execute_complex_plan(plan)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_coordination_metrics(execution_time, success=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Coordination task failed: {e}")
            self._update_coordination_metrics(0, success=False)
            return f"Coordination failed: {str(e)}"
    
    async def can_handle_task(self, task_description: str) -> bool:
        """Coordinator can handle any task by orchestrating other agents"""
        # Coordinator can handle complex tasks that require multiple agents
        complexity = await self.task_planner.analyze_task_complexity(task_description)
        return complexity in [TaskComplexity.COMPLEX, TaskComplexity.ENTERPRISE]
    
    async def _execute_simple_plan(self, plan: TaskPlan) -> str:
        """Execute simple single-agent plan"""
        if not plan.subtasks:
            return "No subtasks to execute"
        
        subtask = plan.subtasks[0]
        
        # Find appropriate agent
        suitable_agents = self.agent_registry.find_agents_by_capability(
            subtask.required_capabilities[0] if subtask.required_capabilities else "general_processing"
        )
        
        if not suitable_agents:
            return f"No suitable agent found for task: {subtask.description}"
        
        # Use the registry's task submission system
        task_request = TaskRequest(
            description=subtask.description,
            requirements=subtask.required_capabilities,
            priority=TaskPriority(subtask.priority),
            context=subtask.context
        )
        
        task_id = await self.agent_registry.submit_task(task_request)
        
        if task_id:
            # Monitor task completion (simplified for now)
            await asyncio.sleep(2)  # Wait for task processing
            return f"Task submitted successfully (ID: {task_id})"
        else:
            return "Failed to submit task to agent"
    
    async def _execute_complex_plan(self, plan: TaskPlan) -> str:
        """Execute complex multi-agent plan"""
        self.logger.info(f"Executing complex plan with {len(plan.subtasks)} subtasks")
        
        if plan.execution_strategy == "sequential":
            return await self._execute_sequential_plan(plan)
        elif plan.execution_strategy == "parallel":
            return await self._execute_parallel_plan(plan)
        else:  # hybrid
            return await self._execute_hybrid_plan(plan)
    
    async def _execute_sequential_plan(self, plan: TaskPlan) -> str:
        """Execute subtasks sequentially"""
        results = []
        previous_result = None
        
        for i, subtask in enumerate(plan.subtasks):
            self.logger.info(f"Executing subtask {i+1}/{len(plan.subtasks)}: {subtask.description}")
            
            # Add previous result to context
            if previous_result:
                subtask.context["previous_result"] = previous_result
            
            # Execute subtask
            result = await self._execute_subtask(subtask)
            results.append(f"Step {i+1}: {result}")
            previous_result = result
            
            # Update plan progress
            plan.progress = (i + 1) / len(plan.subtasks)
            
            # Brief pause between tasks
            await asyncio.sleep(1)
        
        # Compile final result
        final_result = await self._compile_final_result(plan, results)
        plan.final_result = {"compiled_result": final_result}
        plan.status = "completed"
        
        return final_result
    
    async def _execute_parallel_plan(self, plan: TaskPlan) -> str:
        """Execute independent subtasks in parallel"""
        self.logger.info("Executing subtasks in parallel")
        
        # Create tasks for parallel execution
        async_tasks = []
        for subtask in plan.subtasks:
            task_coroutine = self._execute_subtask(subtask)
            async_tasks.append(task_coroutine)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(f"Task {i+1} failed: {str(result)}")
            else:
                processed_results.append(f"Task {i+1}: {result}")
        
        # Compile final result
        final_result = await self._compile_final_result(plan, processed_results)
        plan.final_result = {"compiled_result": final_result}
        plan.status = "completed"
        plan.progress = 1.0
        
        return final_result
    
    async def _execute_hybrid_plan(self, plan: TaskPlan) -> str:
        """Execute plan with mix of parallel and sequential phases"""
        # This is a simplified hybrid approach
        # In practice, you'd analyze the dependency graph more carefully
        
        # For now, execute in phases based on dependencies
        independent_tasks = [st for st in plan.subtasks if not st.dependencies]
        dependent_tasks = [st for st in plan.subtasks if st.dependencies]
        
        results = []
        
        # Execute independent tasks in parallel
        if independent_tasks:
            self.logger.info(f"Executing {len(independent_tasks)} independent tasks in parallel")
            parallel_results = await asyncio.gather(
                *[self._execute_subtask(task) for task in independent_tasks],
                return_exceptions=True
            )
            results.extend([f"Independent task: {r}" for r in parallel_results])
        
        # Execute dependent tasks sequentially
        if dependent_tasks:
            self.logger.info(f"Executing {len(dependent_tasks)} dependent tasks sequentially")
            for task in dependent_tasks:
                result = await self._execute_subtask(task)
                results.append(f"Dependent task: {result}")
        
        # Compile final result
        final_result = await self._compile_final_result(plan, results)
        plan.final_result = {"compiled_result": final_result}
        plan.status = "completed"
        plan.progress = 1.0
        
        return final_result
    
    async def _execute_subtask(self, subtask: SubTask) -> str:
        """Execute individual subtask"""
        try:
            subtask.status = SubTaskStatus.IN_PROGRESS
            subtask.started_at = datetime.now()
            
            # Create task request
            task_request = TaskRequest(
                description=subtask.description,
                requirements=subtask.required_capabilities,
                priority=TaskPriority(subtask.priority),
                context=subtask.context
            )
            
            # Submit to registry
            task_id = await self.agent_registry.submit_task(task_request)
            
            if task_id:
                # For now, simulate waiting for completion
                # In production, you'd monitor the actual task status
                await asyncio.sleep(min(subtask.estimated_duration / 10, 5))  # Scaled down wait
                
                subtask.status = SubTaskStatus.COMPLETED
                subtask.completed_at = datetime.now()
                subtask.result = {"task_id": task_id, "status": "completed"}
                
                return f"Subtask completed successfully (ID: {task_id})"
            else:
                subtask.status = SubTaskStatus.FAILED
                subtask.error_message = "No suitable agent found"
                return "Failed to find suitable agent"
                
        except Exception as e:
            subtask.status = SubTaskStatus.FAILED
            subtask.error_message = str(e)
            self.logger.error(f"Subtask execution failed: {e}")
            return f"Subtask failed: {str(e)}"
    
    async def _compile_final_result(self, plan: TaskPlan, results: List[str]) -> str:
        """Compile final result from all subtask results"""
        if plan.complexity == TaskComplexity.SIMPLE:
            return results[0] if results else "No results"
        
        # Create comprehensive summary
        summary = f"""
**Task Orchestration Complete**

**Original Task:** {plan.original_task}
**Complexity:** {plan.complexity.value}
**Execution Strategy:** {plan.execution_strategy}
**Total Subtasks:** {len(plan.subtasks)}
**Execution Time:** {(datetime.now() - plan.created_at).total_seconds():.1f} seconds

**Results Summary:**
{chr(10).join(f"• {result}" for result in results)}

**Coordination Status:** Successfully orchestrated {len([r for r in results if 'completed' in r.lower()])} out of {len(results)} subtasks.
        """
        
        return summary.strip()
    
    def _update_coordination_metrics(self, execution_time: float, success: bool):
        """Update coordination performance metrics"""
        self.coordination_metrics["total_tasks_orchestrated"] += 1
        
        if success:
            self.coordination_metrics["successful_coordinations"] += 1
        else:
            self.coordination_metrics["failed_coordinations"] += 1
        
        # Update average completion time
        total_tasks = self.coordination_metrics["total_tasks_orchestrated"]
        current_avg = self.coordination_metrics["average_completion_time"]
        self.coordination_metrics["average_completion_time"] = (
            (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        )
    
    def get_active_plans(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active plans"""
        return {
            plan_id: {
                "original_task": plan.original_task,
                "complexity": plan.complexity.value,
                "status": plan.status,
                "progress": plan.progress,
                "subtasks_count": len(plan.subtasks),
                "created_at": plan.created_at.isoformat()
            }
            for plan_id, plan in self.active_plans.items()
        }
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordinator performance metrics"""
        return {
            **self.coordination_metrics,
            "active_plans": len(self.active_plans),
            **self.get_metrics()  # Include base metrics
        }


# Usage example
async def create_coordinator_agent(agent_registry: AgentRegistry) -> CoordinatorAgent:
    """Create and initialize coordinator agent"""
    
    llm_config = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.1
    }
    
    coordinator = CoordinatorAgent(
        agent_registry=agent_registry,
        llm_config=llm_config,
        port=8000
    )
    
    # Initialize the agent
    success = await coordinator.initialize()
    if success:
        logging.info("Coordinator agent initialized successfully")
        return coordinator
    else:
        raise RuntimeError("Failed to initialize coordinator agent")


if __name__ == "__main__":
    # Test the coordinator agent
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    async def test_coordinator():
        from agent_registry import AgentRegistry
        
        # Create registry
        registry = AgentRegistry()
        await registry.start_background_tasks()
        
        # Create coordinator
        coordinator = await create_coordinator_agent(registry)
        
        # Test task processing
        test_context = AgentContext(task_id="test-123")
        result = await coordinator.process_task(
            test_context, 
            "Create a comprehensive analysis report on renewable energy trends"
        )
        
        print(f"Coordination result: {result}")
        print(f"Active plans: {coordinator.get_active_plans()}")
        print(f"Metrics: {coordinator.get_coordination_metrics()}")
        
        await registry.stop_background_tasks()
    
    asyncio.run(test_coordinator())
