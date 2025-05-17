"""
adaptive_rag.py: Adaptive RAG chatbot with multi-PDF support and dynamic routing.

Uploads PDFs, caches embeddings, routes between vectorstore vs. web search,
and uses a LangGraph state machine to generate the best answer.
"""

import os
import tempfile
from typing import List, Optional, TypedDict

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv

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

import logging

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
DEFAULT_TEMPERATURE = 0
CACHE_TTL_SECONDS = 24 * 3600  # 24 hours

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int  # cycle counter to prevent infinite loops


class AdaptiveRagChatbot:
    def __init__(
        self,
        local_llm: str,
        tavily_api_key: str,
        collection_name: str,
        temp_dir: Optional[str] = None,
        max_retries: int = 3
    ):
        self.local_llm = local_llm
        self.tavily_api_key = tavily_api_key
        self.collection_name = collection_name
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_retries = max_retries

        # JSON-output LLM & plain-text LLM
        self.json_llm = ChatOllama(model=self.local_llm, format="json", temperature=DEFAULT_TEMPERATURE)
        self.text_llm = ChatOllama(model=self.local_llm, temperature=DEFAULT_TEMPERATURE)

        self.embeddings = GPT4AllEmbeddings()
        self.web_search_tool = TavilySearchResults(api_key=self.tavily_api_key)

        self.vectorstore = None
        self.retriever = None

        self._build_prompt_chains()
        self.workflow = self._build_workflow_graph().compile()

    def build_vectorstore(self, docs: List[Document]) -> None:
        """Split into overlapping chunks and build a Chroma vectorstore."""
        try:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)
            self.vectorstore = Chroma.from_documents(
                chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name
            )
            self.retriever = self.vectorstore.as_retriever()
        except Exception as e:
            logger.error(f"Failed to build vectorstore: {e}")
            raise

    def _build_prompt_chains(self) -> None:
        """Define LLMChains for routing, grading, rewriting, generating, and usefulness checking."""
        router_pt = PromptTemplate(
            template=(
                "Return JSON {'datasource':'web_search'} or {'datasource':'vectorstore'} "
                "for this question.\nQuestion: {question}"
            ),
            input_variables=["question"]
        )
        self.router_chain = LLMChain(
            llm=self.json_llm, prompt=router_pt, output_parser=JsonOutputParser()
        )

        doc_grade_pt = PromptTemplate(
            template=(
                "Return JSON {'score':'yes'} if this document helps answer the question, "
                "else {'score':'no'}.\nDocument: {document}\nQuestion: {question}"
            ),
            input_variables=["document", "question"]
        )
        self.doc_grader = LLMChain(
            llm=self.json_llm, prompt=doc_grade_pt, output_parser=JsonOutputParser()
        )

        rewrite_pt = PromptTemplate(
            template=(
                "Rewrite the question to improve retrieval quality.\n"
                "Original question: {question}\nRewritten question:"
            ),
            input_variables=["question"]
        )
        self.rewrite_chain = LLMChain(
            llm=self.text_llm, prompt=rewrite_pt, output_parser=StrOutputParser()
        )

        rag_pt = PromptTemplate(
            template=(
                "Answer using these documents.\nDocuments: {documents}\n"
                "Question: {question}\nAnswer:"
            ),
            input_variables=["documents", "question"]
        )
        self.rag_chain = LLMChain(llm=self.text_llm, prompt=rag_pt)

        useful_pt = PromptTemplate(
            template=(
                "Return JSON {'score':'yes'} if this answer is useful, else {'score':'no'}.\n"
                "Answer: {generation}\nQuestion: {question}"
            ),
            input_variables=["generation", "question"]
        )
        self.usefulness_chain = LLMChain(
            llm=self.json_llm, prompt=useful_pt, output_parser=JsonOutputParser()
        )

    def _build_workflow_graph(self) -> StateGraph:
        """Construct the StateGraph with routing and retry logic."""
        graph = StateGraph(GraphState)

        def node_route(s: GraphState) -> str:
            try:
                ds = self.router_chain.run({"question": s["question"]})
                return "web_search" if ds.get("datasource","").lower() == "web_search" else "retrieve"
            except Exception:
                return "retrieve"

        graph.add_node("route_question", node_route)
        graph.set_entry_point("route_question")
        graph.add_conditional_edges(
            "route_question",
            lambda key: key,
            {"retrieve": "retrieve", "web_search": "web_search"}
        )

        def node_retrieve(s: GraphState) -> GraphState:
            if not self.retriever:
                return {**s, "documents": [], "generation": ""}
            docs = self.retriever.get_relevant_documents(s["question"])
            return {**s, "documents": docs}

        graph.add_node("retrieve", node_retrieve)
        graph.add_edge("retrieve", "grade_documents")

        def node_grade_docs(s: GraphState) -> GraphState:
            kept = []
            for d in s["documents"]:
                try:
                    out = self.doc_grader.run({
                        "document": d.page_content,
                        "question": s["question"]
                    })
                    score = out.get("score","").lower()
                except Exception:
                    score = "no"
                if score == "yes":
                    kept.append(d)
            return {**s, "documents": kept}

        graph.add_node("grade_documents", node_grade_docs)
        graph.add_conditional_edges(
            "grade_documents",
            lambda s: "generate" if s["documents"] else "transform_query",
            {"generate": "generate", "transform_query": "transform_query"}
        )

        def node_transform(s: GraphState) -> GraphState:
            new_q = self.rewrite_chain.run({"question": s["question"]})
            return {
                "question": new_q,
                "documents": [],
                "generation": s["generation"],
                "retries": s["retries"] + 1
            }

        graph.add_node("transform_query", node_transform)
        graph.add_edge("transform_query", "route_question")

        def node_web_search(s: GraphState) -> GraphState:
            try:
                hits = self.web_search_tool.invoke(s["question"])
                texts = [h.get("content","") for h in hits or []]
                combined = "\n".join(texts)
                docs = [Document(page_content=combined)] if combined else []
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                docs = []
            return {**s, "documents": docs}

        graph.add_node("web_search", node_web_search)
        graph.add_edge("web_search", "generate")

        def node_generate(s: GraphState) -> GraphState:
            ans = self.rag_chain.run({
                "documents": s["documents"], "question": s["question"]
            })
            return {**s, "generation": ans}

        graph.add_node("generate", node_generate)

        def decide_usefulness(s: GraphState) -> str:
            if s["retries"] >= self.max_retries:
                return END
            try:
                res = self.usefulness_chain.run({
                    "generation": s["generation"], "question": s["question"]
                })
                score = res.get("score","").lower()
            except Exception:
                score = "no"
            return END if score == "yes" else "transform_query"

        graph.add_conditional_edges(
            "generate",
            decide_usefulness,
            {END: END, "transform_query": "transform_query"}
        )

        return graph

    def run(self, question: str) -> str:
        """Execute workflow starting with empty documents. Return final generation."""
        state: GraphState = {
            "question": question,
            "documents": [],
            "generation": "",
            "retries": 0
        }
        for step in self.workflow.stream(state):
            state = list(step.values())[0]
        return state["generation"]


# ---- Streamlit App ---- #

@st.cache_data(
    ttl=CACHE_TTL_SECONDS,
    hash_funcs={UploadedFile: lambda f: (f.name, f.size)}
)
def load_and_split_pdfs(files: List[UploadedFile]) -> List[Document]:
    """Load & split PDFs into Documents, keyed on filename+size."""
    docs: List[Document] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            path = os.path.join(tmpdir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            docs.extend(PyPDFLoader(path).load())
    return docs


@st.cache_resource(
    ttl=CACHE_TTL_SECONDS,
    hash_funcs={Document: lambda d: hash(d.page_content)}
)
def get_bot_and_store(docs: List[Document]) -> AdaptiveRagChatbot:
    """Instantiate chatbot and build vectorstore, keyed on document content."""
    bot = AdaptiveRagChatbot(
        local_llm="llama3",
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        collection_name="adaptive_rag_chroma"
    )
    bot.build_vectorstore(docs)
    return bot


def main():
    st.title("Adaptive RAG ChatBot")
    question = st.text_input("Enter your question:")
    uploaded = st.sidebar.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if st.sidebar.button("Process"):
        if not uploaded:
            st.error("Please upload at least one PDF.")
            return
        if not question.strip():
            st.error("Please enter a non-empty question.")
            return

        docs = load_and_split_pdfs(uploaded)
        st.info(f"Loaded {len(docs)} pages from PDF(s).")

        try:
            bot = get_bot_and_store(docs)
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {e}")
            return

        try:
            answer = bot.run(question)
        except Exception as e:
            st.error(f"Error during generation: {e}")
            return

        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
