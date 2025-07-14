"""Setup script for RagZoom."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ragzoom",
    version="0.1.0",
    description="Incremental, hierarchical RAG memory system",
    author="RagZoom Team",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ragzoom=ragzoom.cli:cli",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)