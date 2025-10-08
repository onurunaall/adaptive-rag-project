#!/bin/bash
set -e

echo "🚀 Setting up InsightEngine development environment..."

# Upgrade pip and install build tools
echo "📦 Upgrading pip and installing build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install the project in editable mode with all dependencies
echo "📚 Installing InsightEngine with all dependencies..."
pip install -e ".[dev,mcp]"

# Verify critical packages
echo "✅ Verifying installations..."
python -c "import langchain; print(f'✓ LangChain: {langchain.__version__}')"
python -c "import streamlit; print(f'✓ Streamlit: {streamlit.__version__}')"
python -c "import chromadb; print(f'✓ ChromaDB: {chromadb.__version__}')"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p chroma_storage
mkdir -p mcp_data/memory_storage
mkdir -p logs

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
else
    echo "✓ .env file found"
fi

# Install pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "🪝 Setting up pre-commit hooks..."
    pre-commit install
fi

echo ""
echo "✨ Setup complete! Your environment is ready."
echo ""
echo "📖 Quick Start:"
echo "  1. Edit .env with your API keys (OPENAI_API_KEY, TAVILY_API_KEY, etc.)"
echo "  2. Run tests: pytest"
echo "  3. Start the app: streamlit run src/main_app.py"
echo ""
echo "🔧 Useful Commands:"
echo "  • Run tests: pytest -v"
echo "  • Start Streamlit: streamlit run src/main_app.py"
echo "  • Check code quality: flake8 src/"
echo "  • Format code: black src/ tests/"
echo ""