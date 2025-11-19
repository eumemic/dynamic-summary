# Build stage for the RagZoom gRPC server
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir and copy only dependency manifests first (speed up build caching)
WORKDIR /app
COPY pyproject.toml poetry.lock* requirements-dev.txt requirements/app.lock requirements/dev.lock ./

# Install pip-tools to sync from lockfiles (keep pip <24.1 for compatibility)
RUN pip install --upgrade 'pip<24.1' setuptools wheel && pip install pip-tools

# Install project dependencies using lock file
RUN if [ -f requirements/dev.lock ]; then pip-sync requirements/dev.lock; \
    else pip install -e .; fi

# Ensure protobuf runtime is available for generated stubs
RUN pip install protobuf

# Copy the rest of the repository
COPY . .

# Install project (and chroma extras for default vector backend) in editable mode
RUN pip install -e '.[chroma]'

# Default environment vars
ENV RAGZOOM_DATABASE_URL=sqlite:////data/sqlite.db \
    PYTHONUNBUFFERED=1

# Create data dir (will be mounted as volume at runtime)
RUN mkdir -p /data

# Expose gRPC port via environment to compose
EXPOSE 50051

CMD ["python", "-m", "ragzoom.cli", "server", "start", "--host", "0.0.0.0", "--port", "50051", "--collect-telemetry"]
