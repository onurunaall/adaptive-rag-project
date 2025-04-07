"""
adaptive_rag.py

Implements a multi-PDF ChatBot using LLama3 and an Adaptive RAG framework.
This module processes uploaded PDFs, builds a vectorstore for retrieval,
constructs several prompt chains (for routing, grading, rewriting, and generation),
and compiles a workflow graph that dynamically selects the best strategy to generate an answer.
"""

import os
import tempfile
import pprint
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict

# Load environment variables from .env file.
load_dotenv()

# --- Import LangChain & LangGraph components ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langgraph.graph import END, StateGraph

# --- Define Graph State Type ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

# --- AdaptiveRagChatbot Class ---
class AdaptiveRagChatbot:
    def __init__(self,
                 local_llm: str,
                 tavily_api_key: str,
                 collection_name: str = "adaptive-rag-chroma",
                 temp_dir: str = None):
        """
        Initialize the chatbot with configuration.
        Parameters:
          - local_llm: Identifier for the local LLM (e.g., "llama3").
          - tavily_api_key: API key for TavilySearch.
          - collection_name: Name for the vectorstore collection.
          - temp_dir: Directory for temporary storage. Defaults to system temp dir.
        """
        self.local_llm = local_llm
        self.tavily_api_key = tavily_api_key
        self.collection_name = collection_name
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Create two shared LLM instances:
        # One for string output (for generation, rewriting, etc.)
        self.str_llm = ChatOllama(model=self.local_llm, format="plain", temperature=0)
        # One for JSON output (for routing, grading, etc.)
        self.json_llm = ChatOllama(model=self.local_llm, format="json", temperature=0)

        # Create a shared embeddings instance.
        self.embedding_instance = GPT4AllEmbeddings()

        # Create a shared web search tool instance.
        self.web_search_tool = TavilySearchResults(k=3, tavily_api_key=self.tavily_api_key)

        # Placeholders for vectorstore and retriever.
        self.vectorstore = None
        self.retriever = None

        # Placeholders for prompt chains.
        self.question_router = None
        self.retrieval_grader = None
        self.rag_chain = None
        self.hallucination_grader = None
        self.answer_grader = None
        self.question_rewriter = None

        # Build prompt chains and compile the workflow graph once.
        self.build_prompt_chains()
        self.compiled_graph = self.build_workflow().compile()

    def process_uploaded_pdfs(self, uploaded_files) -> List[Document]:
        """
        Process uploaded PDF files: save to disk and load their content.
        Returns a list of Document objects.
        Raises an exception if a file fails to load.
        """
        all_docs = []
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        for file in uploaded_files:
            file_path = os.path.join(self.temp_dir, file.name)
            try:
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            except Exception as e:
                raise Exception(f"Error saving file {file.name}: {str(e)}")
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                raise Exception(f"Failed to load {file.name}: {str(e)}")
        return all_docs

    def build_vectorstore(self, docs: List[Document]) -> None:
        """
        Split documents into chunks and build the vectorstore and retriever.
        """
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_chunks = splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(
            documents=doc_chunks,
            collection_name=self.collection_name,
            embedding=self.embedding_instance,
        )
        self.retriever = self.vectorstore.as_retriever()

    def build_prompt_chains(self) -> None:
        """
        Build all prompt chains required for the workflow.
        """
        # Routing chain: returns a JSON with key "datasource".
        routing_prompt = PromptTemplate(
            template=(
                "You are an expert at routing a user question to a vectorstore or web search. "
                "Return a JSON with a single key 'datasource' whose value is either 'web_search' or 'vectorstore'. "
                "Question to route: {question}"
            ),
            input_variables=["question"],
        )
        self.question_router = routing_prompt | self.json_llm | JsonOutputParser()

        # Grading chain: assesses document relevance.
        grading_prompt = PromptTemplate(
            template=(
                "You are a grader assessing the relevance of a document to a question. "
                "Document: {document} "
                "Question: {question} "
                "Return a JSON with a single key 'score' with value 'yes' or 'no'."
            ),
            input_variables=["question", "document"],
        )
        self.retrieval_grader = grading_prompt | self.json_llm | JsonOutputParser()

        # RAG chain: generates an answer (expects plain string output).
        rag_prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = rag_prompt | self.str_llm | StrOutputParser()

        # Hallucination grader: checks if the answer is supported by facts.
        hallucination_prompt = PromptTemplate(
            template=(
                "You are a grader checking if an answer is grounded in the following facts:\n"
                "-------\n{documents}\n-------\n"
                "Answer: {generation}\n"
                "Return a JSON with key 'score' set to 'yes' if supported, 'no' otherwise."
            ),
            input_variables=["documents", "generation"],
        )
        self.hallucination_grader = hallucination_prompt | self.json_llm | JsonOutputParser()

        # Answer usefulness grader.
        answer_prompt = PromptTemplate(
            template=(
                "You are a grader assessing if the following answer is useful for the question.\n"
                "Answer: {generation}\n"
                "Question: {question}\n"
                "Return a JSON with key 'score' set to 'yes' if useful, 'no' otherwise."
            ),
            input_variables=["generation", "question"],
        )
        self.answer_grader = answer_prompt | self.json_llm | JsonOutputParser()

        # Question rewriting chain: improves the question for retrieval.
        rewrite_prompt = PromptTemplate(
            template=(
                "You are a question re-writer. Improve the following question for better vectorstore retrieval.\n"
                "Original question: {question}\n"
                "Improved question:"
            ),
            input_variables=["question"],
        )
        self.question_rewriter = rewrite_prompt | self.str_llm | StrOutputParser()

    def build_workflow(self) -> StateGraph:
        """
        Build and return the workflow graph.
        The flow is:
          - Conditional entry based on routing: either "web_search" or "retrieve"
          - retrieve -> grade_documents -> if no docs, transform_query -> web_search -> retrieve -> generate
          - After generation, decide if answer is supported/useful; if not, loop back to transform_query.
        """
        # Define node functions.
        def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
            print(">>> [AdaptiveRag] Retrieving documents...")
            question = state["question"]
            docs = self.retriever.get_relevant_documents(question)
            return {"documents": docs, "question": question}

        def node_generate(state: Dict[str, Any]) -> Dict[str, Any]:
            print(">>> [AdaptiveRag] Generating answer...")
            question = state["question"]
            docs = state["documents"]
            answer = self.rag_chain.invoke({"context": docs, "question": question})
            return {"documents": docs, "question": question, "generation": answer}

        def node_grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
            print(">>> [AdaptiveRag] Grading documents...")
            question = state["question"]
            docs = state["documents"]
            filtered = []
            for doc in docs:
                result = self.retrieval_grader.invoke({
                    "question": question,
                    "document": doc.page_content
                })
                if result.get("score", "").lower() == "yes":
                    filtered.append(doc)
            return {"documents": filtered, "question": question}

        def node_transform_query(state: Dict[str, Any]) -> Dict[str, Any]:
            print(">>> [AdaptiveRag] Transforming query...")
            question = state["question"]
            new_question = self.question_rewriter.invoke({"question": question})
            return {"documents": state.get("documents", []), "question": new_question}

        def node_web_search(state: Dict[str, Any]) -> Dict[str, Any]:
            print(">>> [AdaptiveRag] Performing web search...")
            question = state["question"]
            docs = state.get("documents", [])
            try:
                results = self.web_search_tool.invoke({"query": question})
                combined = "\n".join([res["content"] for res in results])
                web_doc = Document(page_content=combined)
                docs.append(web_doc)
            except Exception as e:
                print("Error during web search:", str(e))
            return {"documents": docs, "question": question}

        def decide_next(state: Dict[str, Any]) -> str:
            # If no relevant documents remain after grading, transform query.
            if not state.get("documents"):
                print("No relevant documents; transforming query.")
                return "transform_query"
            else:
                print("Relevant documents found; generating answer.")
                return "generate"

        def decide_generation(state: Dict[str, Any]) -> str:
            # After generation, check if the answer is supported by facts.
            generation = state.get("generation", "")
            docs = state.get("documents", [])
            combined_docs = "\n".join([doc.page_content for doc in docs])
            result = self.hallucination_grader.invoke({
                "documents": combined_docs,
                "generation": generation
            })
            if result.get("score", "").lower() == "yes":
                print("Answer is supported; now checking usefulness.")
                result2 = self.answer_grader.invoke({
                    "question": state["question"],
                    "generation": generation
                })
                if result2.get("score", "").lower() == "yes":
                    return "useful"
                else:
                    return "not useful"
            else:
                return "not supported"

        # Build the workflow graph using the defined GraphState type.
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", node_retrieve)
        graph.add_node("grade_documents", node_grade_documents)
        graph.add_node("transform_query", node_transform_query)
        graph.add_node("web_search", node_web_search)
        graph.add_node("generate", node_generate)
        # Entry point: route question based on routing chain.
        def route_question(state: Dict[str, Any]) -> str:
            print(">>> [AdaptiveRag] Routing question...")
            question = state["question"]
            result = self.question_router.invoke({"question": question})
            datasource = result.get("datasource", "").lower()
            if datasource == "web_search":
                print("Routing to web search.")
                return "web_search"
            elif datasource == "vectorstore":
                print("Routing to vectorstore retrieval.")
                return "retrieve"
            else:
                print("Default routing to vectorstore.")
                return "retrieve"
        graph.set_conditional_entry_point(
            route_question,
            {"web_search": "web_search", "vectorstore": "retrieve"}
        )
        graph.add_edge("web_search", "generate")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            decide_next,
            {"transform_query": "transform_query", "generate": "generate"}
        )
        graph.add_edge("transform_query", "retrieve")
        graph.add_conditional_edges(
            "generate",
            decide_generation,
            {"not supported": "generate", "useful": END, "not useful": "transform_query"}
        )
        return graph

    def run_workflow(self, question: str) -> str:
        """
        Run the adaptive RAG workflow with the given question.
        Uses the precompiled graph and returns the final generated answer.
        """
        initial_state = {"question": question}
        final_state = None
        for state in self.compiled_graph.stream(initial_state):
            final_state = state
        return final_state.get("generation", "No generation produced.")

# --- Main Function (Streamlit UI) ---
def main():
    import streamlit as st  # Import here to decouple from core class.
    st.title("Multi-PDF ChatBot using LLAMA3 & Adaptive RAG")
    user_question = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
    uploaded_files = st.sidebar.file_uploader("Upload your PDF file", type=['pdf'], accept_multiple_files=True)
    process = st.sidebar.button("Process")
    if process:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
            st.stop()
        # Instantiate the chatbot with configuration from environment or UI.
        chatbot = AdaptiveRagChatbot(
            local_llm="llama3",
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            collection_name="adaptive-rag-chroma",
            temp_dir=None  # Use system temporary directory.
        )
        try:
            docs = chatbot.process_uploaded_pdfs(uploaded_files)
        except Exception as e:
            st.error(str(e))
            st.stop()
        st.write("PDFs processed successfully.")
        try:
            chatbot.build_vectorstore(docs)
        except Exception as e:
            st.error(f"Error building vectorstore: {str(e)}")
            st.stop()
        # Build prompt chains (graph is compiled in __init__)
        answer = chatbot.run_workflow(user_question)
        st.write("Final Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
