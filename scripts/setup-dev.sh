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
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_ROOT_DIR="$(git rev-parse --show-toplevel)"
    HOOKS_DIR="$(git rev-parse --git-path hooks)"
    mkdir -p "$HOOKS_DIR" 2>/dev/null || true

    # Create pre-commit dispatcher hook (not a symlink)
    # This allows each worktree to use its own version of the hook
    cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Git pre-commit hook dispatcher
# This script calls the worktree's own version of the pre-commit hook
# so each worktree can have its own hook logic

# Find the repository root (this will be the worktree root if in a worktree)
GIT_ROOT="$(git rev-parse --show-toplevel)"

# Check if the worktree has its own pre-commit hook
WORKTREE_HOOK="$GIT_ROOT/scripts/git-hooks/pre-commit"

if [ -f "$WORKTREE_HOOK" ]; then
    # Execute the worktree's own pre-commit hook
    exec "$WORKTREE_HOOK"
else
    # Fallback: no pre-commit hook in this worktree
    echo "No pre-commit hook found at $WORKTREE_HOOK"
    exit 0
fi
EOF
    if chmod +x "$HOOKS_DIR/pre-commit" 2>/dev/null; then
        echo -e "${GREEN}✓ Created pre-commit hook dispatcher at $HOOKS_DIR/pre-commit${NC}"
    else
        echo -e "${YELLOW}⚠️  Could not set executable on $HOOKS_DIR/pre-commit (check permissions)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Not inside a Git worktree; skipping hook setup${NC}"
fi

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

# 3b. Optional Node dev tools (for faster duplicate detection)
echo ""
echo "🧰 Checking Node dev tools (optional)..."
if command -v npm &> /dev/null; then
    if [ -f "$PROJECT_ROOT/package.json" ]; then
        echo "Installing Node dev dependencies (npm install)..."
        if (cd "$PROJECT_ROOT" && npm install --silent --no-fund --no-audit); then
            echo -e "${GREEN}✓ Installed Node dev tools (jscpd)${NC}"
        else
            echo -e "${YELLOW}⚠️  Node dev tools installation did not complete; continuing (optional step)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No package.json found; skipping Node dev tools${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  npm not found; skipping Node dev tools${NC}"
fi

# 4. Set up PostgreSQL with Docker
echo ""
echo "🐘 Setting up PostgreSQL database..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker found${NC}"
    
    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        echo -e "${GREEN}✓ Docker daemon is running${NC}"
        
        # Check if PostgreSQL container already exists
        CONTAINER_NAME="ragzoom-postgres"
        if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "${YELLOW}⚠️  PostgreSQL container already exists${NC}"
            
            # Check if it's running
            if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
                echo -e "${GREEN}✓ PostgreSQL container is running${NC}"
            else
                echo -e "${YELLOW}⚠️  Starting existing PostgreSQL container...${NC}"
                docker start "${CONTAINER_NAME}"
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ PostgreSQL container started${NC}"
                else
                    echo -e "${RED}✗ Failed to start PostgreSQL container${NC}"
                fi
            fi
        else
            echo "Creating PostgreSQL container with pgvector..."
            docker run -d \
                --name "${CONTAINER_NAME}" \
                -e POSTGRES_PASSWORD=postgres \
                -e POSTGRES_DB=ragzoom \
                -p 5432:5432 \
                pgvector/pgvector:pg16
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ PostgreSQL container created and started${NC}"
                echo "   Waiting for PostgreSQL to be ready..."
                
                # Wait for PostgreSQL to be ready
                for i in {1..30}; do
                    if docker exec "${CONTAINER_NAME}" pg_isready -U postgres -d ragzoom &> /dev/null; then
                        echo -e "${GREEN}✓ PostgreSQL is ready!${NC}"
                        break
                    fi
                    if [ $i -eq 30 ]; then
                        echo -e "${YELLOW}⚠️  PostgreSQL may still be starting up${NC}"
                        echo "   Run 'ragzoom doctor' to check status"
                    fi
                    sleep 1
                done
            else
                echo -e "${RED}✗ Failed to create PostgreSQL container${NC}"
                echo "   You may need to set up PostgreSQL manually"
            fi
        fi
    else
        echo -e "${YELLOW}⚠️  Docker is installed but daemon is not running${NC}"
        echo "   Start Docker Desktop or run: sudo systemctl start docker"
    fi
else
    echo -e "${YELLOW}⚠️  Docker not found${NC}"
    echo "   Install Docker Desktop from https://docker.com"
    echo "   Or set RAGZOOM_DATABASE_URL to use existing PostgreSQL"
fi

# 5. Set up environment file
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

# 6. Create necessary directories
echo ""
echo "📁 Setting up directories..."

mkdir -p "$PROJECT_ROOT/logs"
echo -e "${GREEN}✓ Created logs directory${NC}"

# 7. Run tests to verify setup
echo ""
echo "🧪 Verifying tests can be collected..."
if pytest -q --collect-only > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Pytest collection successful${NC}"
else
    echo -e "${YELLOW}⚠️  Pytest collection failed; try ./scripts/run-checks.sh for details${NC}"
fi

# 8. Display helpful information
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
echo "🐘 PostgreSQL (Docker):"
echo "   • Check status: ragzoom doctor"
echo "   • Start container: docker start ragzoom-postgres"
echo "   • Stop container: docker stop ragzoom-postgres"
echo "   • View logs: docker logs ragzoom-postgres"
echo ""
echo "🪝 Git hooks:"
echo "   • Pre-commit dispatcher installed if in a Git worktree"
echo "   • Each worktree can have its own hook logic"
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
