"""
crag.py

Implements a Retrieval-Augmented Generation (RAG) workflow.
Loads a web document, splits it into chunks, creates embeddings and a vectorstore,
and defines a workflow graph that retrieves documents, grades them,
transforms the query if needed, performs a web search when necessary,
and finally generates an answer.
"""

import os
import pprint
from dotenv import load_dotenv
from typing import Dict, Any, TypedDict

# Load environment variables from .env file.
load_dotenv()
# API keys and configuration are now stored in .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# RUN_LOCAL should be "Yes" if using a local LLM; otherwise "No".
RUN_LOCAL = os.getenv("RUN_LOCAL", "No")
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "openai")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Solar")
# Tavily API key is loaded from environment.
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# --- Import LangChain and LangGraph components ---
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate

# --- Define Graph State Type ---
class GraphState(TypedDict):
    keys: Dict[str, Any]

# --- CragWorkflow Class ---
class CragWorkflow:
    def __init__(self):
        # Configuration as instance attributes (can be overridden by env variables)
        self.web_url = os.getenv("WEB_URL", "https://lilianweng.github.io/posts/2023-06-23-agent/")
        self.collection_name = os.getenv("COLLECTION_NAME", "rag-chroma")
        self.run_local = RUN_LOCAL  # "Yes" or "No"
        self.model_choice = MODEL_CHOICE

        # Load the web document.
        self.loader = WebBaseLoader(self.web_url)
        self.docs = self.loader.load()

        # Split the document into smaller chunks.
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=100
        )
        self.chunks = self.splitter.split_documents(self.docs)

        # Select the embedding model.
        if self.run_local == "Yes":
            self.embeddings = GPT4AllEmbeddings()
        elif self.model_choice == "openai":
            self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        else:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY
            )

        # Create the vectorstore and retriever.
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever()
        print("Retriever created:", self.retriever)

    def _get_llm(self):
        """
        Helper method to select and return an LLM instance based on configuration.
        """
        if self.run_local == "Yes":
            return ChatOllama(model=LOCAL_MODEL, temperature=0)
        elif self.model_choice == "openai":
            return ChatOpenAI(model="gpt-4-0125-preview", temperature=0,
                              openai_api_key=OPENAI_API_KEY)
        else:
            return ChatGoogleGenerativeAI(
                model="gemini-pro", google_api_key=GOOGLE_API_KEY,
                convert_system_message_to_human=True, verbose=True
            )

    def retrieve(self, state: GraphState) -> GraphState:
        """
        Retrieve documents from the vectorstore that match the query.
        """
        print(">>> Retrieving documents...")
        data = state["keys"]
        question = data["question"]
        results = self.retriever.get_relevant_documents(question)
        return {"keys": {"documents": results, "question": question}}

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate an answer using a RAG chain.
        """
        print(">>> Generating answer...")
        data = state["keys"]
        question = data["question"]
        docs = data["documents"]

        # Get the prompt template from the hub.
        rag_prompt = hub.pull("rlm/rag-prompt")
        llm = self._get_llm()

        # Build the RAG chain: prompt -> LLM -> output parser.
        rag_chain = rag_prompt | llm | StrOutputParser()
        answer = rag_chain.invoke({"context": docs, "question": question})
        return {"keys": {"documents": docs, "question": question, "generation": answer}}

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grade retrieved documents for relevance to the query.
        If no documents are relevant, mark state to trigger web search.
        """
        print(">>> Grading documents...")
        data = state["keys"]
        question = data["question"]
        docs = data["documents"]

        # Define a grading model.
        class GradeModel(BaseModel):
            score: str = Field(description="Relevance score: 'yes' or 'no'")

        # Create an output parser for grading.
        parser = JsonOutputParser(pydantic_object=GradeModel)
        grade_prompt = PromptTemplate(
            template=(
                "Grade the following document for relevance.\n\n"
                "Document:\n{context}\n\n"
                "Question: {question}\n\n"
                "If relevant, return 'yes'; else return 'no'.\n"
                "Use these instructions: {format_instructions}"
            ),
            input_variables=["question", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        llm = self._get_llm()
        grading_chain = grade_prompt | llm | parser

        filtered_docs = []
        for doc in docs:
            result = grading_chain.invoke({
                "question": question,
                "context": doc.page_content,
                "format_instructions": parser.get_format_instructions(),
            })
            if result["score"].lower() == "yes":
                filtered_docs.append(doc)

        # If no relevant documents, mark state to trigger web search.
        run_web_search = "Yes" if len(filtered_docs) == 0 else "No"
        return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": run_web_search}}

    def transform_query(self, state: GraphState) -> GraphState:
        """
        Refine the query to improve search results.
        """
        print(">>> Transforming query...")
        data = state["keys"]
        question = data["question"]
        docs = data.get("documents", [])

        # Create a prompt to refine the query.
        rewrite_prompt = PromptTemplate(
            template=(
                "Refine the following query for better search results.\n\n"
                "Original query:\n{question}\n\n"
                "Refined query:"
            ),
            input_variables=["question"],
        )
        llm = self._get_llm()
        rewrite_chain = rewrite_prompt | llm | StrOutputParser()
        new_query = rewrite_chain.invoke({"question": question})
        return {"keys": {"documents": docs, "question": new_query}}

    def web_search(self, state: GraphState) -> GraphState:
        """
        Perform a web search using Tavily API and add the results as a new document.
        """
        print(">>> Performing web search...")
        data = state["keys"]
        question = data["question"]
        docs = data.get("documents", [])
        try:
            search_tool = TavilySearchResults()
            results = search_tool.invoke({"query": question})
            combined_text = "\n".join([res["content"] for res in results])
            web_doc = Document(page_content=combined_text)
            docs.append(web_doc)
        except Exception as error:
            # Catch more specific exceptions as needed.
            print("Error during web search:", error)
        return {"keys": {"documents": docs, "question": question}}

    def decide_next_step(self, state: GraphState) -> str:
        """
        Decide the next node based on graded documents.
        If no relevant documents, first transform the query, then do web search.
        If there are relevant documents, go directly to generate.
        """
        data = state["keys"]
        run_web = data.get("run_web_search", "No")
        if run_web == "Yes":
            return "transform_query"
        else:
            return "generate"

    def build_workflow(self) -> StateGraph:
        """
        Build and compile the workflow graph.
        Flow:
          retrieve -> grade_documents -> (if docs) generate; (if no docs) transform_query -> web_search -> generate
        """
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_next_step,
            {"transform_query": "transform_query", "generate": "generate"}
        )
        # If the query is transformed, perform web search then generate.
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        return workflow

# --- Example Execution ---
if __name__ == "__main__":
    # Create an instance of the workflow.
    crag = CragWorkflow()
    workflow = crag.build_workflow()
    # Set up an initial state with a sample query.
    initial_state = {"keys": {"question": "Explain how agent memory works."}}
    for output in workflow.stream(initial_state):
        for node, state_val in output.items():
            print("Node:", node)
            # Uncomment to see detailed state:
            # pprint.pprint(state_val["keys"], indent=2)
        print("-----")
    # Print the final generated answer.
    pprint.pprint(output["keys"].get("generation", "No generation produced."))
