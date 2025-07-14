#!/bin/bash
# Development environment setup script for RagZoom

echo "🚀 Setting up RagZoom development environment..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to create symlink with error handling
create_symlink() {
    local source="$1"
    local target="$2"
    
    # Remove existing file/symlink if it exists
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo -e "${YELLOW}Removing existing $(basename "$target")...${NC}"
        rm -f "$target"
    fi
    
    # Create symlink
    ln -s "$source" "$target"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Linked $(basename "$target")${NC}"
    else
        echo -e "${RED}✗ Failed to link $(basename "$target")${NC}"
        return 1
    fi
}

# 1. Set up Git hooks
echo ""
echo "📎 Setting up Git hooks..."

# Create .git/hooks directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/.git/hooks"

# Symlink pre-commit hook
create_symlink "$PROJECT_ROOT/scripts/git-hooks/pre-commit" "$PROJECT_ROOT/.git/hooks/pre-commit"

# Symlink pre-push hook
create_symlink "$PROJECT_ROOT/scripts/git-hooks/pre-push" "$PROJECT_ROOT/.git/hooks/pre-push"

# 2. Check Python environment
echo ""
echo "🐍 Checking Python environment..."

if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  No virtual environment detected${NC}"
    echo "   Consider creating one with: python -m venv venv"
    echo "   And activating it with: source venv/bin/activate"
else
    echo -e "${GREEN}✓ Virtual environment active: $VIRTUAL_ENV${NC}"
fi

# 3. Install dependencies
echo ""
echo "📦 Installing dependencies..."

if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    pip install -q -r "$PROJECT_ROOT/requirements.txt"
    echo -e "${GREEN}✓ Installed base requirements${NC}"
fi

if [ -f "$PROJECT_ROOT/requirements-dev.txt" ]; then
    pip install -q -r "$PROJECT_ROOT/requirements-dev.txt"
    echo -e "${GREEN}✓ Installed dev requirements${NC}"
fi

# Install package in development mode
pip install -q -e "$PROJECT_ROOT"
echo -e "${GREEN}✓ Installed RagZoom in development mode${NC}"

# 4. Set up environment file
echo ""
echo "🔐 Checking environment configuration..."

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo -e "${YELLOW}⚠️  Created .env from .env.example${NC}"
        echo "   Please add your OPENAI_API_KEY to .env"
    else
        echo -e "${YELLOW}⚠️  No .env file found${NC}"
        echo "   Create one with: echo 'OPENAI_API_KEY=your-key-here' > .env"
    fi
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# 5. Create necessary directories
echo ""
echo "📁 Setting up directories..."

mkdir -p "$PROJECT_ROOT/logs"
echo -e "${GREEN}✓ Created logs directory${NC}"

# 6. Run tests to verify setup
echo ""
echo "🧪 Running tests to verify setup..."

# Run just a quick test to verify everything is working
pytest "$PROJECT_ROOT/tests/test_utils.py" -q
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Tests are working${NC}"
else
    echo -e "${RED}✗ Tests failed - check your setup${NC}"
fi

# 7. Display helpful information
echo ""
echo "✨ Setup complete!"
echo ""
echo "📚 Quick reference:"
echo "   • Run tests: pytest tests/ -v"
echo "   • Quick tests: ./test_quick.sh"
echo "   • Format code: black ragzoom/ tests/"
echo "   • Lint code: ruff check ragzoom/ tests/"
echo "   • Type check: mypy ragzoom/"
echo ""
echo "🪝 Git hooks installed:"
echo "   • pre-commit: Runs relevant tests for changed files"
echo "   • pre-push: Runs full test suite before pushing"
echo "   • Skip hooks with --no-verify flag"
echo ""
echo "📖 Documentation:"
echo "   • Testing strategy: docs/testing-strategy.md"
echo "   • Project info: CLAUDE.md"
echo ""

# Check if OPENAI_API_KEY is set
if [ -f "$PROJECT_ROOT/.env" ]; then
    if grep -q "OPENAI_API_KEY=your-key-here\|OPENAI_API_KEY=$" "$PROJECT_ROOT/.env" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Don't forget to add your OpenAI API key to .env!${NC}"
    fi
fi