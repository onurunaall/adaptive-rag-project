"""
adaptive_rag.py

Adaptive RAG chatbot with multi-PDF support and dynamic routing.
"""

import os
import tempfile
import logging
from typing import List, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TEMPERATURE = 0
CACHE_TTL = 24 * 3600  # 24 hours

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int  # Prevent infinite loops

class AdaptiveRagChatbot:
    """
    Orchestrates adaptive retrieval-augmented generation using LangGraph and LangChain.
    """

    def __init__(self,
                 model_name: str,
                 web_search_key: str,
                 vectorstore_name: str,
                 temp_dir: Optional[str] = None,
                 max_retries: int = 3):

        self.model_name = model_name
        self.web_search_key = web_search_key
        self.vectorstore_name = vectorstore_name
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_retries = max_retries

        self.json_llm = ChatOllama(model=self.model_name,
                                   format="json",
                                   temperature=TEMPERATURE)
        self.text_llm = ChatOllama(model=self.model_name,
                                   temperature=TEMPERATURE)

        # Embedding and search tools
        self.embeddings = GPT4AllEmbeddings()
        self.search_tool = TavilySearchResults(api_key=self.web_search_key)

        # Vectorstore & retriever - set by build_vectorstore()
        self.vectorstore = None
        self.retriever = None

        # Build prompt chains and the workflow graph
        self._create_chains()
        self.workflow = self._create_graph().compile()

    def build_vectorstore(self, docs: List[Document]) -> None:
        """
        Split docs and build Chroma vectorstore.
        Falls back to non-tiktoken splitter if needed.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                      chunk_overlap=CHUNK_OVERLAP)
        
        except ImportError:
            logger.warning("tiktoken not installed; using fallback splitter.")
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                      chunk_overlap=CHUNK_OVERLAP)
        
        except Exception as e:
            logger.warning(f"Error with tiktoken splitter ({e}); using fallback.")
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                      chunk_overlap=CHUNK_OVERLAP)

        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(chunks,
                                                 embedding=self.embeddings,
                                                 collection_name=self.vectorstore_name)
        
        self.retriever = self.vectorstore.as_retriever()

    def _create_chains(self) -> None:
        """
        Define all prompt chains used in workflow.
        """
        # Routing (vectorstore vs web)
        self.router = LLMChain(llm=self.json_llm,
                               prompt=PromptTemplate(template=(
                                   "Return JSON {'datasource':'web_search'} or {'datasource':'vectorstore'}.\n"
                                   "Question: {question}"),
                                   input_variables=["question"]),
                               output_parser=JsonOutputParser())

        # Document relevance grader
        self.doc_grader = LLMChain(llm=self.json_llm,
                                   prompt=PromptTemplate(template=(
                                       "Return JSON {'score':'yes'} if this document is relevant, else {'score':'no'}.\n"
                                       "Document: {document}\nQuestion: {question}"),
                                       input_variables=["document", "question"]),
                                   output_parser=JsonOutputParser())

        # Question rewriter
        self.rewriter = LLMChain(llm=self.text_llm,
                                 prompt=PromptTemplate(template=(
                                     "Rewrite the question to improve retrieval.\n"
                                     "Original question: {question}\nRewritten question:"),
                                     input_variables=["question"]),
                                 output_parser=StrOutputParser())

        # RAG generator
        self.generator = LLMChain(llm=self.text_llm,
                                  prompt=PromptTemplate(template=(
                                      "Answer using these documents.\n"
                                      "Documents: {documents}\nQuestion: {question}\nAnswer:"),
                                      input_variables=["documents", "question"]))

        # Usefulness checker
        self.checker = LLMChain(llm=self.json_llm,
                                prompt=PromptTemplate(template=(
                                    "Return JSON {'score':'yes'} if this answer is useful, else {'score':'no'}.\n"
                                    "Answer: {generation}\nQuestion: {question}"),
                                    input_variables=["generation", "question"]),
                                output_parser=JsonOutputParser())

    def _create_graph(self) -> StateGraph:
        """
        Create the LangGraph workflow with nodes and edge logic.
        """
        graph = StateGraph(GraphState)

        # Node: Route question
        graph.add_node("route_question", self._route_question_node)
        graph.set_entry_point("route_question")
        graph.add_conditional_edges(source="route_question",
                                   condition=self._route_condition,
                                   mapping={"retrieve": "retrieve",
                                            "web_search": "web_search"})

        # Node: Vectorstore retrieval
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_edge("retrieve", "grade_documents")

        # Node: Document grading
        graph.add_node("grade_documents", self._grade_documents_node)
        graph.add_conditional_edges(source="grade_documents",
                                   condition=self._grade_condition,
                                   mapping={"generate": "generate",
                                            "transform_query": "transform_query"})

        # Node: Rewrite query
        graph.add_node("transform_query", self._transform_query_node)
        graph.add_edge("transform_query", "route_question")

        # Node: Web search fallback
        graph.add_node("web_search", self._web_search_node)
        graph.add_edge("web_search", "generate")

        # Node: RAG generation
        graph.add_node("generate", self._generate_node)
        graph.add_conditional_edges(source="generate",
                                   condition=self._decide_useful_node,
                                   mapping={END: END,
                                            "transform_query": "transform_query"})

        return graph

    def _route_question_node(self, state: GraphState) -> str:
        """
        Decide whether to use vectorstore or web search.
        """
        try:
            result = self.router.run({"question": state["question"]})
            choice = result.get("datasource", "").lower()
            if choice == "web_search":
                return "web_search"
        except Exception:
            logger.info("Router failed, defaulting to 'retrieve'")
        return "retrieve"

    def _route_condition(self, decision: str) -> str:
        return decision

    def _retrieve_node(self, state: GraphState) -> GraphState:
        """
        Retrieve docs from the vectorstore.
        """
        if self.retriever is None:
            return {**state, "documents": []}
        
        docs = self.retriever.get_relevant_documents(state["question"])
        return {**state, "documents": docs}

    def _grade_documents_node(self, state: GraphState) -> GraphState:
        """
        Keep only docs graded as relevant.
        """
        kept: List[Document] = []
        
        for doc in state["documents"]:
            try:
                out = self.doc_grader.run({"document": doc.page_content,
                                           "question": state["question"]})
                score = out.get("score", "").lower()
            except Exception:
                score = "no"
            
            if score == "yes":
                kept.append(doc)
        
        return {**state, "documents": kept}

    def _grade_condition(self, state: GraphState) -> str:
        """
        Continue to generate if docs remain, else try query rewriting.
        """
        return "generate" if state["documents"] else "transform_query"

    def _transform_query_node(self, state: GraphState) -> GraphState:
        """
        Rewrite the question, reset previous outputs, increment retry counter.
        """
        new_q = self.rewriter.run({"question": state["question"]})
        return {
            "question": new_q,
            "documents": [],
            "generation": "",
            "retries": state["retries"] + 1}

    def _web_search_node(self, state: GraphState) -> GraphState:
        """
        Use web search, treat each result as a Document.
        """
        try:
            hits = self.search_tool.invoke(state["question"])
            docs = []
            for hit in hits or []:
                text = hit.get("content", "")
                
                if text:
                    docs.append(Document(page_content=text))
        
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            docs = []
        
        return {**state, "documents": docs}
        
    def _generate_node(self, state: GraphState) -> GraphState:
        """
        Use LLM to generate answer from relevant docs.
        """
        answer = self.generator.run({"documents": state["documents"],
                                     "question": state["question"]})
        
        return {**state, "generation": answer}

    def _decide_useful_node(self, state: GraphState) -> str:
        """
        Is the answer useful? If not, retry unless out of retries.
        """
        if state["retries"] >= self.max_retries:
            return END
        
        try:
            out = self.checker.run({"generation": state["generation"],
                                    "question": state["question"]})
            score = out.get("score", "").lower()
        
        except Exception:
            score = "no"
        
        return END if score == "yes" else "transform_query"

    def run(self, question: str) -> str:
        """
        Run the workflow, return the final answer string.
        """
        state: GraphState = {"question": question,
                             "documents": [],
                             "generation": "",
                             "retries": 0}
        
        for step in self.workflow.stream(state):
            state = next(iter(step.values()))
        
        return state["generation"]

@st.cache_data(ttl=CACHE_TTL, hash_funcs={UploadedFile: lambda f: (f.name, f.size)})
def load_pdfs(files: List[UploadedFile]) -> List[Document]:
    """
    Load and split uploaded PDFs into Document pages.
    """
    docs: List[Document] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for file in files:
            path = os.path.join(tmpdir, file.name)
            
            with open(path, "wb") as out:
                out.write(file.getbuffer())
            
            docs.extend(PyPDFLoader(path).load())
    
    return docs

@st.cache_resource(ttl=CACHE_TTL, hash_funcs={Document: lambda d: hash(d.page_content)})
def init_chatbot(docs: List[Document],
                 model_name: str,
                 collection_name: str) -> AdaptiveRagChatbot:
    """
    Initialize the chatbot and build its vectorstore.
    """
    bot = AdaptiveRagChatbot(model_name=model_name,
                             web_search_key=os.getenv("TAVILY_API_KEY", ""),
                             vectorstore_name=collection_name)
    
    bot.build_vectorstore(docs)
    return bot

def main():
    st.title("Adaptive RAG ChatBot")

    # Sidebar: config
    st.sidebar.header("Configuration")
    model_name = st.sidebar.text_input("LLM model name",
                                       value=os.getenv("LLAMA_MODEL_NAME", "llama3"))
    
    collection_name = st.sidebar.text_input("Chroma collection name",
                                            value=os.getenv("VECTORSTORE_NAME", "adaptive_rag_chroma"))
    
    uploads = st.sidebar.file_uploader("Upload PDFs",
                                       type="pdf",
                                       accept_multiple_files=True)

    question = st.text_input("Enter your question:")

    if st.sidebar.button("Process"):
        if not uploads:
            st.error("Please upload at least one PDF.")
            return
        
        if not question.strip():
            st.error("Please enter a non-empty question.")
            return

        docs = load_pdfs(uploads)
        st.info(f"Loaded {len(docs)} pages from PDF(s).")

        try:
            bot = init_chatbot(docs, model_name, collection_name)
        except Exception as e:
            st.error(f"Initialization error: {e}")
            return

        try:
            answer = bot.run(question)
        except Exception as e:
            st.error(f"Generation error: {e}")
            return

        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
