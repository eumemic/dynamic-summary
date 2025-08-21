# Development Scripts

This directory contains scripts and tools for RagZoom development.

## setup-dev.sh

Sets up the complete development environment including:
- Git hooks (pre-commit and pre-push)
- Python dependencies
- Environment configuration
- Directory structure
- Verification tests

### Usage

```bash
# From the project root
./scripts/setup-dev.sh

# Or from anywhere
cd /path/to/dynamic-summary
bash scripts/setup-dev.sh
```

### What it does

1. **Installs Git Hooks**
   - Symlinks pre-commit hook (runs relevant tests on commit)
   - Symlinks pre-push hook (runs full test suite before push)

2. **Sets up Python Environment**
   - Installs requirements.txt and requirements-dev.txt
   - Installs RagZoom in development mode (`pip install -e .`)

3. **Configuration**
   - Creates .env from .env.example if needed
   - Creates necessary directories (logs/)

4. **Verification**
   - Runs a quick test to ensure everything is working

## git-hooks/

Contains the actual git hook scripts that are symlinked into `.git/hooks/`:

- **pre-commit**: Runs only the tests relevant to changed files (fast feedback)
- **pre-push**: Runs the complete test suite before pushing to remote


## Adding New Scripts

When adding new development scripts:
1. Place them in this directory
2. Make them executable: `chmod +x script-name.sh`
3. Document them in this README
4. Consider adding them to setup-dev.sh if they should run during setup