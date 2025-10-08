#!/bin/bash
# Startup verification script for GitHub Codespaces
# This runs after the container is created to verify everything works

set -e

echo "🔍 Running startup verification checks..."
echo ""

# Check Python version
echo "✓ Python version:"
python --version

# Check pip version
echo "✓ Pip version:"
pip --version

# Verify critical packages
echo ""
echo "📦 Verifying critical packages..."

packages=(
    "langchain"
    "streamlit"
    "chromadb"
    "openai"
    "pydantic"
    "langchain_openai"
)

for package in "${packages[@]}"; do
    if python -c "import ${package}" 2>/dev/null; then
        version=$(python -c "import ${package}; print(${package}.__version__)" 2>/dev/null || echo "version unavailable")
        echo "  ✓ ${package}: ${version}"
    else
        echo "  ✗ ${package}: NOT INSTALLED"
        exit 1
    fi
done

# Check for .env file
echo ""
echo "🔐 Checking configuration..."
if [ -f .env ]; then
    echo "  ✓ .env file exists"
    
    # Check for required keys (without showing values)
    if grep -q "OPENAI_API_KEY=" .env && ! grep -q 'OPENAI_API_KEY=""' .env; then
        echo "  ✓ OPENAI_API_KEY is set"
    else
        echo "  ⚠️  OPENAI_API_KEY not configured"
        echo "     Edit .env and add your OpenAI API key"
    fi
    
    if grep -q "TAVILY_API_KEY=" .env && ! grep -q 'TAVILY_API_KEY=""' .env; then
        echo "  ✓ TAVILY_API_KEY is set"
    else
        echo "  ⚠️  TAVILY_API_KEY not configured (optional)"
    fi
else
    echo "  ✗ .env file not found!"
    echo "     Creating from template..."
    cp .env.example .env
    echo "     Please edit .env and add your API keys"
fi

# Check directory structure
echo ""
echo "📁 Checking directories..."
directories=(
    "src"
    "tests"
    "mcp"
    "chroma_storage"
    "logs"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ ${dir}/"
    else
        echo "  ✗ ${dir}/ not found"
    fi
done

# Run a quick import test
echo ""
echo "🧪 Running import test..."
if python -c "from src.core_rag_engine import CoreRAGEngine; print('  ✓ CoreRAGEngine imports successfully')" 2>/dev/null; then
    echo "  ✓ Core imports working"
else
    echo "  ✗ Import test failed"
    echo "  Try: make clean && make install-dev"
    exit 1
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Startup verification complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. Configure API keys in .env (if not done)"
echo "   2. Run tests: make test"
echo "   3. Start app: make run"
echo ""
echo "💡 Quick commands:"
echo "   • make help      - Show all commands"
echo "   • make run       - Start Streamlit app"
echo "   • make test      - Run test suite"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"