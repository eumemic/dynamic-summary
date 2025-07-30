"""Setup script for RagZoom Telemetry."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ragzoom-telemetry",
    version="0.1.0",
    description="Developer tools for analyzing RagZoom telemetry data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RagZoom Team",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ragzoom-telemetry=ragzoom_telemetry.cli:cli",
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