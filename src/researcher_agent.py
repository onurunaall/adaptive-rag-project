import asyncio
import json
import uuid
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, urljoin

from fastmcp import FastMCP
from mcp.types import Tool, TextContent

from a2a.types import AgentSkill

from .base_agent import BaseAgent, AgentRole, AgentStatus, AgentCapability, AgentContext
from .core_rag_engine import CoreRAGEngine
from .stock import fetch_stock_news_documents
from .scraper import scrape_urls_as_documents

class ResearchScope(str, Enum):
    """Research scope and depth levels"""
    QUICK_LOOKUP = "quick_lookup"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    REAL_TIME = "real_time"
    HISTORICAL = "historical"

class InformationSource(str, Enum):
    """Types of information sources"""
    INTERNAL_RAG = "internal_rag"
    WEB_SEARCH = "web_search"
    ACADEMIC = "academic"
    NEWS = "news"
    FINANCIAL = "financial"
    SOCIAL = "social"
    TECHNICAL = "technical"
    GOVERNMENT = "government"

@dataclass
class ResearchQuery:
    """Structured research query with parameters"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    scope: ResearchScope = ResearchScope.STANDARD
    sources: List[InformationSource] = field(default_factory=list)
    time_range: Optional[Tuple[datetime, datetime]] = None
    max_sources: int = 10
    priority_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ResearchResult:
    """Research finding with metadata"""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: InformationSource
    source_url: Optional[str] = None
    source_title: str = ""
    content: str = ""
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_facts: List[str] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)

@dataclass
class ResearchReport:
    """Comprehensive research compilation"""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: ResearchQuery
    results: List[ResearchResult] = field(default_factory=list)
    synthesis: str = ""
    confidence_level: float = 0.0
    sources_consulted: int = 0
    research_duration: float = 0.0  # seconds
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class ResearchEngine:
    """Enhanced research engine that integrates multiple information sources"""
    
    def __init__(
        self, 
        core_rag_engine: CoreRAGEngine,
        llm_config: Dict[str, Any] = None
    ):
        self.core_rag_engine = core_rag_engine
        self.llm_config = llm_config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Research capabilities
        self.search_strategies = {
            ResearchScope.QUICK_LOOKUP: self._quick_lookup_strategy,
            ResearchScope.STANDARD: self._standard_research_strategy,
            ResearchScope.COMPREHENSIVE: self._comprehensive_research_strategy,
            ResearchScope.REAL_TIME: self._real_time_research_strategy,
            ResearchScope.HISTORICAL: self._historical_research_strategy
        }
        
        # Quality filters
        self.credibility_indicators = {
            "high": [".edu", ".gov", ".org", "scholar.google", "pubmed", "arxiv"],
            "medium": [".com", "reuters", "bloomberg", "wsj", "nytimes"],
            "low": ["blog", "forum", "social", "personal"]
        }
    
    async def conduct_research(self, query: ResearchQuery) -> ResearchReport:
        """Conduct comprehensive research based on query parameters"""
        self.logger.info(f"Starting research: {query.question}")
        start_time = datetime.now()
        
        # Select and execute research strategy
        strategy = self.search_strategies.get(query.scope, self._standard_research_strategy)
        results = await strategy(query)
        
        # Filter and rank results
        filtered_results = await self._filter_and_rank_results(results, query)
        
        # Synthesize findings
        synthesis = await self._synthesize_findings(filtered_results, query)
        
        # Calculate research metrics
        research_duration = (datetime.now() - start_time).total_seconds()
        confidence_level = self._calculate_confidence_level(filtered_results)
        
        # Extract key insights
        key_findings = await self._extract_key_findings(filtered_results)
        recommendations = await self._generate_recommendations(filtered_results, query)
        
        report = ResearchReport(
            original_query=query,
            results=filtered_results,
            synthesis=synthesis,
            confidence_level=confidence_level,
            sources_consulted=len(filtered_results),
            research_duration=research_duration,
            key_findings=key_findings,
            recommendations=recommendations
        )
        
        self.logger.info(f"Research completed in {research_duration:.1f}s with {len(filtered_results)} sources")
        return report
    
    async def _quick_lookup_strategy(self, query: ResearchQuery) -> List[ResearchResult]:
        """Quick lookup - prioritize internal RAG and fast sources"""
        results = []
        
        # First check internal RAG system
        if InformationSource.INTERNAL_RAG in query.sources or not query.sources:
            rag_results = await self._search_internal_rag(query.question, max_results=3)
            results.extend(rag_results)
        
        # If insufficient results, add web search
        if len(results) < 2 and (InformationSource.WEB_SEARCH in query.sources or not query.sources):
            web_results = await self._search_web(query.question, max_results=2)
            results.extend(web_results)
        
        return results[:query.max_sources]
    
    async def _standard_research_strategy(self, query: ResearchQuery) -> List[ResearchResult]:
        """Standard research - balanced approach across multiple sources"""
        results = []
        
        # Internal knowledge base first
        if not query.sources or InformationSource.INTERNAL_RAG in query.sources:
            rag_results = await self._search_internal_rag(query.question, max_results=5)
            results.extend(rag_results)
        
        # Web search for broader context
        if not query.sources or InformationSource.WEB_SEARCH in query.sources:
            web_results = await self._search_web(query.question, max_results=5)
            results.extend(web_results)
        
        # News for current information
        if InformationSource.NEWS in query.sources:
            news_results = await self._search_news(query.question, max_results=3)
            results.extend(news_results)
        
        # Financial data if relevant
        if InformationSource.FINANCIAL in query.sources:
            financial_results = await self._search_financial_data(query.question, max_results=3)
            results.extend(financial_results)
        
        return results[:query.max_sources]
    
    async def _comprehensive_research_strategy(self, query: ResearchQuery) -> List[ResearchResult]:
        """Comprehensive research - exhaustive multi-source approach"""
        all_results = []
        
        # Parallel searches across all available sources
        search_tasks = []
        
        # Internal RAG
        search_tasks.append(self._search_internal_rag(query.question, max_results=8))
        
        # Web search with multiple approaches
        search_tasks.append(self._search_web(query.question, max_results=10))
        search_tasks.append(self._search_academic_sources(query.question, max_results=5))
        
        # News and current events
        search_tasks.append(self._search_news(query.question, max_results=5))
        
        # Financial data if applicable
        if self._is_financial_query(query.question):
            search_tasks.append(self._search_financial_data(query.question, max_results=5))
        
        # Technical documentation if applicable
        if self._is_technical_query(query.question):
            search_tasks.append(self._search_technical_docs(query.question, max_results=3))
        
        # Execute all searches concurrently
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine results
        for result_set in search_results:
            if isinstance(result_set, list):
                all_results.extend(result_set)
            elif isinstance(result_set, Exception):
                self.logger.error(f"Search task failed: {result_set}")
        
        return all_results[:query.max_sources * 2]  # Allow more results for comprehensive research
    
    async def _real_time_research_strategy(self, query: ResearchQuery) -> List[ResearchResult]:
        """Real-time research - focus on current and live data"""
        results = []
        
        # Prioritize news and current events
        news_results = await self._search_news(query.question, max_results=8, time_filter="24h")
        results.extend(news_results)
        
        # Financial markets if relevant
        if self._is_financial_query(query.question):
            financial_results = await self._search_financial_data(query.question, max_results=5)
            results.extend(financial_results)
        
        # Social trends (simplified - would integrate with social media APIs)
        if InformationSource.SOCIAL in query.sources:
            social_results = await self._search_social_trends(query.question, max_results=3)
            results.extend(social_results)
        
        # Web search with recency filter
        web_results = await self._search_web(query.question, max_results=5, time_filter="week")
        results.extend(web_results)
        
        return results[:query.max_sources]
    
    async def _historical_research_strategy(self, query: ResearchQuery) -> List[ResearchResult]:
        """Historical research - focus on trends and time-series data"""
        results = []
        
        # Internal historical data
        rag_results = await self._search_internal_rag(query.question, max_results=8)
        results.extend(rag_results)
        
        # Academic and scholarly sources
        academic_results = await self._search_academic_sources(query.question, max_results=5)
        results.extend(academic_results)
        
        # Government and official sources
        if InformationSource.GOVERNMENT in query.sources:
            gov_results = await self._search_government_sources(query.question, max_results=3)
            results.extend(gov_results)
        
        # Financial historical data
        if self._is_financial_query(query.question):
            historical_financial = await self._search_financial_data(query.question, max_results=5, historical=True)
            results.extend(historical_financial)
        
        return results[:query.max_sources]
    
    # Source-specific search methods
    async def _search_internal_rag(self, question: str, max_results: int = 5) -> List[ResearchResult]:
        """Search internal RAG system"""
        try:
            # Use your existing CoreRAGEngine
            response = self.core_rag_engine.run_full_rag_workflow(question)
            
            results = []
            for i, source in enumerate(response.get("sources", [])[:max_results]):
                result = ResearchResult(
                    source=InformationSource.INTERNAL_RAG,
                    source_url=source.get("source", "internal_document"),
                    source_title=f"Internal Document {i+1}",
                    content=source.get("preview", ""),
                    relevance_score=0.9,
                    credibility_score=1.0,
                    metadata={"source_type": "internal_rag", "rag_response": response.get("answer", "")}
                )
                results.append(result)
            
            # Also include the main RAG answer
            if response.get("answer"):
                main_result = ResearchResult(
                    source=InformationSource.INTERNAL_RAG,
                    source_title="RAG System Analysis",
                    content=response["answer"],
                    relevance_score=1.0,
                    credibility_score=1.0,
                    metadata={"source_type": "rag_synthesis"}
                )
                results.insert(0, main_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Internal RAG search failed: {e}")
            return []
    
    async def _search_web(self, question: str, max_results: int = 5, time_filter: str = None) -> List[ResearchResult]:
        """Search web using scraping capabilities"""
        try:
            # Use your existing scraper functionality
            search_urls = self._generate_search_urls(question, max_results)
            
            if not search_urls:
                return []
            
            # Scrape the URLs
            documents = scrape_urls_as_documents(
                urls=search_urls,
                user_goal_for_scraping=f"Research: {question}"
            )
            
            results = []
            for doc in documents:
                # Calculate relevance based on keyword matching
                relevance = self._calculate_relevance_score(question, doc.page_content)
                credibility = self._calculate_credibility_score(doc.metadata.get("source", ""))
                
                result = ResearchResult(
                    source=InformationSource.WEB_SEARCH,
                    source_url=doc.metadata.get("source"),
                    source_title=self._extract_title_from_content(doc.page_content),
                    content=doc.page_content[:1000],  # Limit content length
                    relevance_score=relevance,
                    credibility_score=credibility,
                    metadata=doc.metadata
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return []
    
    async def _search_news(self, question: str, max_results: int = 5, time_filter: str = None) -> List[ResearchResult]:
        """Search news sources"""
        try:
            # Use news-specific URLs or APIs
            news_urls = self._generate_news_search_urls(question, max_results, time_filter)
            
            documents = scrape_urls_as_documents(
                urls=news_urls,
                user_goal_for_scraping=f"News research: {question}"
            )
            
            results = []
            for doc in documents:
                relevance = self._calculate_relevance_score(question, doc.page_content)
                
                result = ResearchResult(
                    source=InformationSource.NEWS,
                    source_url=doc.metadata.get("source"),
                    source_title=self._extract_title_from_content(doc.page_content),
                    content=doc.page_content[:800],
                    relevance_score=relevance,
                    credibility_score=0.8,
                    timestamp=datetime.now(),
                    metadata={**doc.metadata, "source_type": "news"}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"News search failed: {e}")
            return []
    
    async def _search_financial_data(self, question: str, max_results: int = 5, historical: bool = False) -> List[ResearchResult]:
        """Search financial data sources"""
        try:
            # Extract potential ticker symbols from question
            tickers = self._extract_tickers_from_query(question)
            
            if not tickers:
                return []
            
            # Use your existing stock news functionality
            documents = fetch_stock_news_documents(
                tickers_input=tickers,
                max_articles_per_ticker=max_results // len(tickers) + 1
            )
            
            results = []
            for doc in documents:
                relevance = self._calculate_relevance_score(question, doc.page_content)
                
                result = ResearchResult(
                    source=InformationSource.FINANCIAL,
                    source_url=doc.metadata.get("source"),
                    source_title=doc.metadata.get("title", "Financial News"),
                    content=doc.page_content,
                    relevance_score=relevance,
                    credibility_score=0.9,
                    timestamp=self._parse_timestamp(doc.metadata.get("published_date")),
                    metadata={**doc.metadata, "tickers": tickers}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Financial data search failed: {e}")
            return []
    
    async def _search_academic_sources(self, question: str, max_results: int = 5) -> List[ResearchResult]:
        """Search academic and scholarly sources"""
        # This would integrate with academic databases
        # For now, return a placeholder structure
        return []
    
    async def _search_technical_docs(self, question: str, max_results: int = 3) -> List[ResearchResult]:
        """Search technical documentation"""
        # This would search developer docs, API references, etc.
        return []
    
    async def _search_government_sources(self, question: str, max_results: int = 3) -> List[ResearchResult]:
        """Search government and official sources"""
        # This would search .gov sites and official databases
        return []
    
    async def _search_social_trends(self, question: str, max_results: int = 3) -> List[ResearchResult]:
        """Search social media trends"""
        # This would integrate with social media APIs
        return []
    
    # Utility methods
    def _generate_search_urls(self, question: str, max_results: int) -> List[str]:
        """Generate search URLs for web scraping"""
        # This is simplified - in practice you'd use search APIs or construct better URLs
        import urllib.parse
        
        query = urllib.parse.quote_plus(question)
        
        # Example search URLs (you'd customize these)
        base_urls = [
            f"https://www.google.com/search?q={query}",
            f"https://duckduckgo.com/?q={query}",
        ]
        
        # For demo purposes, return some example URLs
        # In practice, you'd parse search results to get actual content URLs
        return []
    
    def _generate_news_search_urls(self, question: str, max_results: int, time_filter: str = None) -> List[str]:
        """Generate news-specific search URLs"""
        # Similar to above but for news sources
        return []
    
    def _extract_tickers_from_query(self, question: str) -> List[str]:
        """Extract stock ticker symbols from query"""
        # Look for uppercase 3-5 letter combinations that might be tickers
        import re
        potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', question)
        
        # Common tickers for testing
        common_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA']
        return [ticker for ticker in potential_tickers if ticker in common_tickers]
    
    def _calculate_relevance_score(self, question: str, content: str) -> float:
        """Calculate relevance score based on keyword matching"""
        question_words = set(question.lower().split())
        content_words = set(content.lower().split())
        
        # Simple Jaccard similarity
        intersection = question_words.intersection(content_words)
        union = question_words.union(content_words)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_credibility_score(self, source_url: str) -> float:
        """Calculate source credibility based on URL"""
        if not source_url:
            return 0.5
        
        url_lower = source_url.lower()
        
        # Check credibility indicators
        for level, indicators in self.credibility_indicators.items():
            for indicator in indicators:
                if indicator in url_lower:
                    if level == "high":
                        return 0.9
                    elif level == "medium":
                        return 0.7
                    else:
                        return 0.4
        
        return 0.6  # Default moderate credibility
    
    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from content"""
        lines = content.split('\n')
        for line in lines:
            if line.strip() and len(line) < 200:
                return line.strip()[:100]
        return "Untitled Document"
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp from various formats"""
        if not timestamp_str:
            return None
        
        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%B %d, %Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _is_financial_query(self, question: str) -> bool:
        """Check if query is finance-related"""
        financial_keywords = [
            "stock", "market", "price", "trading", "investment", "financial",
            "earnings", "revenue", "profit", "company", "ticker"
        ]
        return any(keyword in question.lower() for keyword in financial_keywords)
    
    def _is_technical_query(self, question: str) -> bool:
        """Check if query is technical/programming related"""
        technical_keywords = [
            "api", "code", "programming", "software", "development", "technical",
            "documentation", "framework", "library", "algorithm"
        ]
        return any(keyword in question.lower() for keyword in technical_keywords)
    
    async def _filter_and_rank_results(self, results: List[ResearchResult], query: ResearchQuery) -> List[ResearchResult]:
        """Filter and rank results based on quality and relevance"""
        if not results:
            return []
        
        # Filter out low-quality results
        filtered = [
            result for result in results
            if result.relevance_score > 0.1 and result.credibility_score > 0.2
        ]
        
        # Apply keyword filters
        if query.exclude_keywords:
            filtered = [
                result for result in filtered
                if not any(keyword.lower() in result.content.lower() for keyword in query.exclude_keywords)
            ]
        
        # Boost results with priority keywords
        if query.priority_keywords:
            for result in filtered:
                boost = sum(
                    0.2 for keyword in query.priority_keywords
                    if keyword.lower() in result.content.lower()
                )
                result.relevance_score = min(1.0, result.relevance_score + boost)
        
        # Sort by combined score (relevance * credibility)
        filtered.sort(key=lambda r: r.relevance_score * r.credibility_score, reverse=True)
        
        return filtered[:query.max_sources]
    
    async def _synthesize_findings(self, results: List[ResearchResult], query: ResearchQuery) -> str:
        """Synthesize findings into coherent summary"""
        if not results:
            return "No relevant information found for the given query."
        
        # For now, create a structured summary
        # In practice, you'd use an LLM to synthesize the content
        
        synthesis_parts = [
            f"**Research Summary for:** {query.question}\n",
            f"**Sources Consulted:** {len(results)}",
            f"**Research Scope:** {query.scope.value}\n"
        ]
        
        # Group by source type
        by_source = {}
        for result in results:
            if result.source not in by_source:
                by_source[result.source] = []
            by_source[result.source].append(result)
        
        for source_type, source_results in by_source.items():
            synthesis_parts.append(f"\n**{source_type.value.title()} Sources ({len(source_results)}):**")
            
            for result in source_results[:3]:  # Limit to top 3 per source
                synthesis_parts.append(f"• {result.source_title}")
                # Add key excerpt
                excerpt = result.content[:200] + "..." if len(result.content) > 200 else result.content
                synthesis_parts.append(f"  {excerpt}")
        
        return "\n".join(synthesis_parts)
    
    def _calculate_confidence_level(self, results: List[ResearchResult]) -> float:
        """Calculate overall confidence level for research"""
        if not results:
            return 0.0
        
        # Consider number of sources, credibility, and relevance
        source_diversity = len(set(result.source for result in results))
        avg_credibility = sum(result.credibility_score for result in results) / len(results)
        avg_relevance = sum(result.relevance_score for result in results) / len(results)
        
        # Weight factors
        diversity_score = min(1.0, source_diversity / 3)  # Normalize to max 3 source types
        
        confidence = (diversity_score * 0.3 + avg_credibility * 0.4 + avg_relevance * 0.3)
        return min(1.0, confidence)
    
    async def _extract_key_findings(self, results: List[ResearchResult]) -> List[str]:
        """Extract key findings from research results"""
        findings = []
        
        # Simple keyword-based extraction for now
        # In practice, you'd use NLP or LLM for better extraction
        
        for result in results[:5]:  # Focus on top results
            # Look for sentences with key indicators
            sentences = result.content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(indicator in sentence.lower() for indicator in 
                    ['found that', 'shows that', 'indicates', 'according to', 'research shows']):
                    findings.append(sentence)
        
        return findings[:5]  # Return top 5 findings
    
    async def _generate_recommendations(self, results: List[ResearchResult], query: ResearchQuery) -> List[str]:
        """Generate actionable recommendations based on findings"""
        recommendations = []
        
        # Simple recommendation generation based on research scope
        if query.scope == ResearchScope.COMPREHENSIVE:
            recommendations.append("Consider conducting follow-up research on specific subtopics identified")
            recommendations.append("Verify findings with additional authoritative sources")
        
        if len(results) < query.max_sources // 2:
            recommendations.append("Expand search terms or consider alternative keywords")
            recommendations.append("Include additional information sources in future research")
        
        source_types = set(result.source for result in results)
        if InformationSource.ACADEMIC not in source_types:
            recommendations.append("Consider including academic sources for deeper analysis")
        
        return recommendations

class ResearcherAgent(BaseAgent):
    """
    Advanced researcher agent that integrates with your existing InsightEngine
    and provides enhanced information gathering capabilities.
    """
    
    def __init__(
        self,
        core_rag_engine: CoreRAGEngine,
        llm_config: Optional[Dict[str, Any]] = None,
        port: int = 8001
    ):
        # Define researcher capabilities
        capabilities = [
            AgentCapability(
                name="information_gathering",
                description="Comprehensive information gathering from multiple sources",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "scope": {"type": "string", "enum": [s.value for s in ResearchScope]},
                        "sources": {"type": "array", "items": {"type": "string"}}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "report": {"type": "object"},
                        "confidence": {"type": "number"},
                        "sources_count": {"type": "integer"}
                    }
                }
            ),
            AgentCapability(
                name="rag_search",
                description="Search internal knowledge base using RAG",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}}
            ),
            AgentCapability(
                name="web_research",
                description="Research current information from web sources",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"findings": {"type": "array"}}}
            ),
            AgentCapability(
                name="financial_research",
                description="Research financial and market information",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"financial_data": {"type": "object"}}}
            ),
            AgentCapability(
                name="fact_verification",
                description="Verify facts and claims across multiple sources",
                input_schema={"type": "object", "properties": {"claim": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"verification": {"type": "object"}}}
            )
        ]
        
        super().__init__(
            name="Researcher Agent",
            role=AgentRole.RESEARCHER,
            description="Advanced information gathering agent with RAG integration and multi-source research capabilities",
            capabilities=capabilities,
            llm_config=llm_config,
            port=port
        )
        
        # Researcher-specific components
        self.core_rag_engine = core_rag_engine
        self.research_engine = ResearchEngine(core_rag_engine, llm_config)
        self.research_history: List[ResearchReport] = []
        
        # Performance tracking
        self.research_metrics = {
            "queries_processed": 0,
            "successful_researches": 0,
            "average_sources_per_query": 0,
            "average_confidence_level": 0.0,
            "total_research_time": 0.0
        }
    
    async def _get_role_specific_skills(self) -> List[AgentSkill]:
        """Define researcher-specific skills"""
        return [
            AgentSkill(
                name="comprehensive_research",
                description="Conduct comprehensive research using multiple information sources",
                capabilities=["information_gathering", "web_research", "rag_search"],
                examples=[
                    "Research market trends for renewable energy sector",
                    "Gather information on latest AI developments",
                    "Investigate competitive landscape for fintech companies"
                ]
            ),
            AgentSkill(
                name="fact_checking",
                description="Verify facts and claims using authoritative sources",
                capabilities=["fact_verification", "information_gathering"],
                examples=[
                    "Verify financial metrics for public companies",
                    "Cross-reference scientific claims with peer-reviewed sources",
                    "Validate historical data and statistics"
                ]
            ),
            AgentSkill(
                name="financial_intelligence",
                description="Research financial markets, companies, and economic trends",
                capabilities=["financial_research", "information_gathering"],
                examples=[
                    "Analyze stock performance and market conditions",
                    "Research company financial health and prospects",
                    "Track economic indicators and trends"
                ]
            ),
            AgentSkill(
                name="real_time_monitoring",
                description="Monitor current events and breaking information",
                capabilities=["web_research", "information_gathering"],
                examples=[
                    "Track breaking news on specific topics",
                    "Monitor social media trends and sentiment",
                    "Follow real-time market movements"
                ]
            )
        ]
    
    async def process_task(self, task_context: AgentContext, message: str) -> str:
        """Process research task"""
        try:
            self.logger.info(f"Processing research task: {message}")
            start_time = datetime.now()
            
            # Parse research parameters from context
            research_scope = ResearchScope(
                task_context.metadata.get("scope", ResearchScope.STANDARD.value)
            )
            
            max_sources = task_context.metadata.get("max_sources", 10)
            source_types = task_context.metadata.get("sources", [])
            
            # Create research query
            query = ResearchQuery(
                question=message,
                scope=research_scope,
                sources=[InformationSource(s) for s in source_types] if source_types else [],
                max_sources=max_sources,
                context=task_context.metadata
            )
            
            # Conduct research
            report = await self.research_engine.conduct_research(query)
            
            # Store research history
            self.research_history.append(report)
            if len(self.research_history) > 100:  # Limit history size
                self.research_history.pop(0)
            
            # Update metrics
            research_time = (datetime.now() - start_time).total_seconds()
            self._update_research_metrics(report, research_time)
            
            # Format response
            response = self._format_research_response(report)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Research task failed: {e}")
            return f"Research failed: {str(e)}"
    
    async def can_handle_task(self, task_description: str) -> bool:
        """Check if this agent can handle the research task"""
        research_keywords = [
            "research", "find", "gather", "information", "search", "investigate",
            "analyze", "data", "facts", "sources", "verify", "check",
            "news", "trends", "market", "financial", "stock", "company"
        ]
        
        return any(keyword in task_description.lower() for keyword in research_keywords)
    
    def _format_research_response(self, report: ResearchReport) -> str:
        """Format research report for response"""
        response_parts = [
            f"# Research Report: {report.original_query.question}\n",
            f"**Research Completed:** {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Sources Consulted:** {report.sources_consulted}",
            f"**Confidence Level:** {report.confidence_level:.1%}",
            f"**Research Duration:** {report.research_duration:.1f} seconds\n"
        ]
        
        # Add synthesis
        if report.synthesis:
            response_parts.append("## Summary")
            response_parts.append(report.synthesis)
            response_parts.append("")
        
        # Add key findings
        if report.key_findings:
            response_parts.append("## Key Findings")
            for i, finding in enumerate(report.key_findings, 1):
                response_parts.append(f"{i}. {finding}")
            response_parts.append("")
        
        # Add recommendations
        if report.recommendations:
            response_parts.append("## Recommendations")
            for i, rec in enumerate(report.recommendations, 1):
                response_parts.append(f"{i}. {rec}")
            response_parts.append("")
        
        # Add top sources
        if report.results:
            response_parts.append("## Top Sources")
            for i, result in enumerate(report.results[:5], 1):
                response_parts.append(f"{i}. **{result.source_title}**")
                if result.source_url:
                    response_parts.append(f"   Source: {result.source_url}")
                response_parts.append(f"   Relevance: {result.relevance_score:.1%} | Credibility: {result.credibility_score:.1%}")
                response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _update_research_metrics(self, report: ResearchReport, research_time: float):
        """Update research performance metrics"""
        self.research_metrics["queries_processed"] += 1
        self.research_metrics["successful_researches"] += 1
        self.research_metrics["total_research_time"] += research_time
        
        # Update averages
        total_queries = self.research_metrics["queries_processed"]
        
        # Average sources per query
        current_avg_sources = self.research_metrics["average_sources_per_query"]
        self.research_metrics["average_sources_per_query"] = (
            (current_avg_sources * (total_queries - 1) + report.sources_consulted) / total_queries
        )
        
        # Average confidence level
        current_avg_confidence = self.research_metrics["average_confidence_level"]
        self.research_metrics["average_confidence_level"] = (
            (current_avg_confidence * (total_queries - 1) + report.confidence_level) / total_queries
        )
    
    def get_research_metrics(self) -> Dict[str, Any]:
        """Get research performance metrics"""
        return {
            **self.research_metrics,
            "recent_research_count": len(self.research_history),
            **self.get_metrics()  # Include base agent metrics
        }
    
    def get_recent_research(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent research reports summary"""
        recent = self.research_history[-limit:]
        return [
            {
                "report_id": report.report_id,
                "query": report.original_query.question,
                "sources_consulted": report.sources_consulted,
                "confidence_level": report.confidence_level,
                "created_at": report.created_at.isoformat()
            }
            for report in recent
        ]


# Usage example
async def create_researcher_agent(core_rag_engine: CoreRAGEngine) -> ResearcherAgent:
    """Create and initialize researcher agent"""
    
    llm_config = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.1
    }
    
    researcher = ResearcherAgent(
        core_rag_engine=core_rag_engine,
        llm_config=llm_config,
        port=8001
    )
    
    # Initialize the agent
    success = await researcher.initialize()
    if success:
        logging.info("Researcher agent initialized successfully")
        return researcher
    else:
        raise RuntimeError("Failed to initialize researcher agent")


if __name__ == "__main__":
    # Test the researcher agent
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    async def test_researcher():
        from core_rag_engine import CoreRAGEngine
        
        # Create RAG engine
        rag_engine = CoreRAGEngine()
        
        # Create researcher
        researcher = await create_researcher_agent(rag_engine)
        
        # Test research
        test_context = AgentContext(
            task_id="research-test-123",
            metadata={"scope": "comprehensive", "max_sources": 8}
        )
        
        result = await researcher.process_task(
            test_context,
            "Research the latest developments in artificial intelligence and machine learning"
        )
        
        print(f"Research result:\n{result}")
        print(f"\nMetrics: {researcher.get_research_metrics()}")
    
    asyncio.run(test_researcher())
