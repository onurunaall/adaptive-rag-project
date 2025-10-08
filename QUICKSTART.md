# Quick Start Guide

Get up and running with Adaptive RAG Engine in under 5 minutes!

## 📋 Prerequisites

- Python 3.12 or higher
- pip package manager
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/adaptive-rag-project.git
cd adaptive-rag-project
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Replace 'your-api-key-here' with your actual key
```

Your `.env` file should look like:
```bash
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
```

## 🎯 Your First Query

### Option 1: Using the Streamlit UI (Recommended)

```bash
# Run the application
make run

# Or directly:
streamlit run src/main_app.py
```

Then open http://localhost:8501 in your browser!

**Try these steps in the UI:**
1. Click "Ingest Documents" in the sidebar
2. Upload a few text files or PDFs
3. Enter a question in the main chat
4. Watch the magic happen! ✨

### Option 2: Using Python API

Create a file named `test_rag.py`:

```python
from src.core_rag_engine import CoreRAGEngine
from langchain_core.documents import Document

# Initialize the engine
engine = CoreRAGEngine()

# Create some sample documents
documents = [
    Document(
        page_content="Paris is the capital of France. It is famous for the Eiffel Tower.",
        metadata={"source": "geography.txt"}
    ),
    Document(
        page_content="Python is a high-level programming language created by Guido van Rossum.",
        metadata={"source": "programming.txt"}
    ),
    Document(
        page_content="Machine learning is a subset of artificial intelligence.",
        metadata={"source": "ai.txt"}
    )
]

# Ingest the documents
print("📚 Ingesting documents...")
engine.ingest(
    direct_documents=documents,
    collection_name="quickstart_demo",
    recreate_collection=True
)
print("✅ Documents ingested successfully!")

# Ask a question
print("\n❓ Asking question...")
result = engine.run_full_rag_workflow(
    question="What is the capital of France?",
    collection_name="quickstart_demo"
)

# Print the answer
print("\n💡 Answer:", result["answer"])
print("\n📄 Retrieved", len(result["documents"]), "documents")
```

Run it:
```bash
python test_rag.py
```

Expected output:
```
📚 Ingesting documents...
✅ Documents ingested successfully!

❓ Asking question...
💡 Answer: The capital of France is Paris.

📄 Retrieved 3 documents
```

## 📚 Common Use Cases

### Use Case 1: Question Answering from Documents

```python
from src.core_rag_engine import CoreRAGEngine

engine = CoreRAGEngine()

# Ingest a directory of documents
engine.ingest(
    source_directory="./my_documents",
    collection_name="my_knowledge_base"
)

# Ask questions
result = engine.run_full_rag_workflow(
    question="What is our company policy on remote work?",
    collection_name="my_knowledge_base"
)

print(result["answer"])
```

### Use Case 2: Web-Enhanced Search

```python
# Enable web search (requires Tavily API key in .env)
engine = CoreRAGEngine(
    tavily_api_key="your-tavily-key"
)

# If local knowledge is insufficient, automatically searches web
result = engine.run_full_rag_workflow(
    question="What are the latest developments in quantum computing?",
    collection_name="tech_docs"
)
```

### Use Case 3: Code Documentation Search

```python
# Ingest your codebase
engine.ingest(
    source_directory="./src",
    collection_name="codebase",
    chunking_strategy="adaptive"  # Automatically detects code
)

# Query about your code
result = engine.run_full_rag_workflow(
    question="How does the caching system work?",
    collection_name="codebase"
)
```

## 🔧 Configuration Tips

### Choosing the Right Model

For **best quality**:
```bash
LLM_MODEL_NAME=gpt-4o
```

For **faster responses** (recommended for getting started):
```bash
LLM_MODEL_NAME=gpt-4o-mini
```

For **local/private deployment**:
```bash
LLM_PROVIDER=ollama
LLM_MODEL_NAME=llama2
# Requires Ollama installed locally
```

### Adjusting Chunk Size

For **technical documents**:
```bash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

For **short-form content**:
```bash
CHUNK_SIZE=300
CHUNK_OVERLAP=50
```

### Enabling Advanced Features

```bash
# Enable hybrid search (better accuracy)
ENABLE_HYBRID_SEARCH=true

# Enable web search fallback
ENABLE_WEB_SEARCH=true
TAVILY_API_KEY=your-key-here

# Increase grounding attempts for better accuracy
MAX_GROUNDING_ATTEMPTS=5
```

## 🧪 Testing Your Setup

Run the test suite to verify everything works:

```bash
# Run all tests
make test

# Expected output:
# ========== 50 passed in 90.00s ==========
```

If tests fail, check:
1. ✅ API key is correctly set in `.env`
2. ✅ Virtual environment is activated
3. ✅ All dependencies are installed

## 📖 Next Steps

### Learn More

- 📘 [Full Documentation](README.md)
- 🏗️ [Architecture Guide](ARCHITECTURE.md)
- 🤝 [Contributing Guide](CONTRIBUTING.md)
- 📝 [API Reference](README.md#-api-reference)

### Try Advanced Features

1. **Custom Chunking Strategies**
   ```python
   from src.config import EngineSettings
   
   settings = EngineSettings(
       chunking_strategy="adaptive",
       chunk_size=1000
   )
   engine = CoreRAGEngine(engine_settings=settings)
   ```

2. **Streaming Responses**
   ```python
   for chunk in engine.stream_answer("What is RAG?"):
       print(chunk, end="", flush=True)
   ```

3. **Multi-Collection Search**
   ```python
   # Create multiple knowledge bases
   engine.ingest(docs1, collection_name="tech_docs")
   engine.ingest(docs2, collection_name="business_docs")
   
   # Query each separately
   tech_answer = engine.run_full_rag_workflow(
       "What is kubernetes?",
       collection_name="tech_docs"
   )
   ```

## ❓ Troubleshooting

### "OPENAI_API_KEY is missing"

**Solution:**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# If empty, add to .env file
echo "OPENAI_API_KEY=sk-your-key" >> .env

# Restart the application
```

### "ModuleNotFoundError"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### "Collection not found"

**Solution:**
```python
# List available collections
engine.list_collections()

# Create a new collection
engine.ingest(
    direct_documents=docs,
    collection_name="new_collection",
    recreate_collection=True
)
```

### Slow Queries

**Solution:**
```bash
# Enable caching in .env
ENABLE_CACHING=true
CACHE_TTL_SECONDS=600

# Use smaller model for faster responses
LLM_MODEL_NAME=gpt-4o-mini
```

## 💡 Pro Tips

### 1. Start Small
Begin with a small document collection (10-20 documents) to understand the system before scaling up.

### 2. Use Appropriate Models
- **gpt-4o-mini**: Fast, cost-effective, good for most use cases
- **gpt-4o**: Best quality for complex queries
- **ollama**: For complete privacy and local deployment

### 3. Optimize Chunking
Experiment with `CHUNK_SIZE` based on your document type:
- Code: 500-800 characters
- Articles: 800-1200 characters
- Technical docs: 1000-1500 characters

### 4. Monitor Cache Performance
```python
# Check cache statistics
stats = engine.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}")
print(f"Memory usage: {stats['estimated_memory_mb']} MB")
```

### 5. Use Collections Effectively
Create separate collections for different domains:
- `product_docs` - Product documentation
- `company_policies` - Internal policies
- `customer_data` - Customer information
- `technical_docs` - Technical documentation

## 🎓 Example Projects

### Project 1: Personal Knowledge Base

```python
# Ingest your personal notes
engine.ingest(
    source_directory="~/Documents/notes",
    collection_name="personal_kb"
)

# Ask questions about your notes
result = engine.run_full_rag_workflow(
    "What were my key takeaways from the machine learning course?",
    collection_name="personal_kb"
)
```

### Project 2: Customer Support Bot

```python
# Ingest support documentation
engine.ingest(
    source_directory="./support_docs",
    collection_name="support_kb"
)

# Answer customer queries
def answer_customer_query(query: str) -> str:
    result = engine.run_full_rag_workflow(
        question=query,
        collection_name="support_kb"
    )
    return result["answer"]

# Use in your support system
answer = answer_customer_query("How do I reset my password?")
```

### Project 3: Research Assistant

```python
# Ingest research papers
from src.scraper_feed import scrape_urls_as_documents

papers = scrape_urls_as_documents([
    "https://arxiv.org/abs/2005.11401",
    "https://arxiv.org/abs/2307.09288"
])

engine.ingest(
    direct_documents=papers,
    collection_name="research_papers"
)

# Query your research collection
result = engine.run_full_rag_workflow(
    "What are the main challenges in retrieval-augmented generation?",
    collection_name="research_papers"
)
```

## 🎉 Congratulations!

You're now ready to build amazing RAG applications! 

**Join our community:**
- 💬 [GitHub Discussions](https://github.com/yourusername/adaptive-rag-project/discussions)
- 🐛 [Report Issues](https://github.com/yourusername/adaptive-rag-project/issues)
- 📧 [Email Support](mailto:support@yourproject.com)

Happy building! 🚀