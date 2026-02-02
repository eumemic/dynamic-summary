#!/usr/bin/env python3
"""Setup script for RagZoom OpenClaw integration.

Run from the dynamic-summary repo root:
    python integrations/openclaw/scripts/setup.py

Or with options:
    python integrations/openclaw/scripts/setup.py --openai-key sk-... --port 50052
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"  → {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def find_repo_root() -> Path:
    """Find the dynamic-summary repo root."""
    # Check if we're in the repo
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() and "ragzoom" in (cwd / "pyproject.toml").read_text():
        return cwd
    
    # Check parent directories
    for parent in cwd.parents:
        if (parent / "pyproject.toml").exists():
            try:
                if "ragzoom" in (parent / "pyproject.toml").read_text():
                    return parent
            except:
                pass
    
    # Check common locations
    for path in [
        Path.home() / "code" / "dynamic-summary",
        Path.home() / "projects" / "dynamic-summary",
        Path("/Users/jarvis/code/dynamic-summary"),
    ]:
        if path.exists() and (path / "pyproject.toml").exists():
            return path
    
    return None


def check_python_version() -> bool:
    """Check Python version is 3.12+."""
    if sys.version_info < (3, 12):
        print(f"❌ Python 3.12+ required, found {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def setup_venv(repo_root: Path) -> Path:
    """Create or verify venv exists."""
    venv_path = repo_root / ".venv"
    
    if venv_path.exists():
        print(f"✅ Venv exists at {venv_path}")
    else:
        print(f"Creating venv at {venv_path}...")
        run([sys.executable, "-m", "venv", str(venv_path)])
        print(f"✅ Created venv")
    
    return venv_path


def get_venv_python(venv_path: Path) -> Path:
    """Get path to venv Python."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def install_packages(repo_root: Path, venv_python: Path) -> bool:
    """Install ragzoom and openclaw integration."""
    print("\nInstalling packages...")
    
    # Upgrade pip
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    
    # Install main package
    print("Installing ragzoom...")
    result = run([str(venv_python), "-m", "pip", "install", "-e", str(repo_root)], check=False)
    if result.returncode != 0:
        print("❌ Failed to install ragzoom")
        return False
    
    # Install openclaw integration
    openclaw_path = repo_root / "integrations" / "openclaw"
    if openclaw_path.exists():
        print("Installing ragzoom-openclaw...")
        result = run([str(venv_python), "-m", "pip", "install", "-e", str(openclaw_path)], check=False)
        if result.returncode != 0:
            print("❌ Failed to install ragzoom-openclaw")
            return False
    
    print("✅ Packages installed")
    return True


def setup_config(openai_key: str | None) -> bool:
    """Set up the config directory and .env file."""
    config_dir = Path.home() / ".config" / "ragzoom"
    env_file = config_dir / ".env"
    
    config_dir.mkdir(parents=True, exist_ok=True)
    
    if env_file.exists():
        content = env_file.read_text()
        if "OPENAI_API_KEY" in content:
            print(f"✅ Config exists at {env_file}")
            return True
    
    if not openai_key:
        print(f"\n⚠️  OpenAI API key needed for embeddings.")
        print(f"   Create {env_file} with:")
        print(f"   OPENAI_API_KEY=sk-your-key-here")
        return False
    
    env_file.write_text(f"OPENAI_API_KEY={openai_key}\n")
    print(f"✅ Created config at {env_file}")
    return True


def verify_install(venv_path: Path) -> bool:
    """Verify the installation works."""
    print("\nVerifying installation...")
    
    venv_bin = venv_path / "bin"
    
    # Check ragzoom CLI
    ragzoom_cli = venv_bin / "ragzoom"
    if not ragzoom_cli.exists():
        print("❌ ragzoom CLI not found")
        return False
    print("✅ ragzoom CLI available")
    
    # Check openclaw CLI
    openclaw_cli = venv_bin / "ragzoom-openclaw"
    if not openclaw_cli.exists():
        print("❌ ragzoom-openclaw CLI not found")
        return False
    print("✅ ragzoom-openclaw CLI available")
    
    return True


def print_next_steps(repo_root: Path, venv_path: Path, port: int):
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print(f"""
1. Start the RagZoom server:
   cd {repo_root}
   source .venv/bin/activate
   ragzoom server start --port {port}

2. Sync your OpenClaw session:
   ragzoom-openclaw sync ~/.openclaw/agents/main/sessions/<session-id>.jsonl --document-id <your-name>-main

3. Test a query (Python):
   from ragzoom_claude_code.recall import execute_recall, format_for_cli
   result = execute_recall("test query", document_id="<your-name>-main", server_address="localhost:{port}")
   print(format_for_cli(result))
""")


def main():
    parser = argparse.ArgumentParser(description="Set up RagZoom OpenClaw integration")
    parser.add_argument("--openai-key", help="OpenAI API key for embeddings")
    parser.add_argument("--port", type=int, default=50052, help="Server port (default: 50052)")
    parser.add_argument("--repo", type=Path, help="Path to dynamic-summary repo")
    args = parser.parse_args()
    
    print("🚀 RagZoom OpenClaw Setup\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Find repo root
    repo_root = args.repo or find_repo_root()
    if not repo_root:
        print("❌ Could not find dynamic-summary repo")
        print("   Run from repo directory or use --repo /path/to/dynamic-summary")
        sys.exit(1)
    print(f"✅ Repo found at {repo_root}")
    
    # Set up venv
    venv_path = setup_venv(repo_root)
    venv_python = get_venv_python(venv_path)
    
    # Install packages
    if not install_packages(repo_root, venv_python):
        sys.exit(1)
    
    # Set up config
    setup_config(args.openai_key)
    
    # Verify
    if not verify_install(venv_path):
        sys.exit(1)
    
    # Print next steps
    print_next_steps(repo_root, venv_path, args.port)


if __name__ == "__main__":
    main()
