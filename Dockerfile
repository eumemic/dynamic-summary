# Build stage for the RagZoom gRPC server
FROM python:3.11-slim AS base

# Install system dependencies (including libpq for psycopg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpq-dev \
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

# Install project with postgres support
RUN pip install -e '.[chroma]' && pip install 'psycopg[binary]'

# Default environment vars (Railway will override DATABASE_URL)
ENV PYTHONUNBUFFERED=1 \
    RAGZOOM_BACKEND=postgres

# Create data dir (will be mounted as volume at runtime)
RUN mkdir -p /data

# Expose gRPC port (Railway overrides via PORT env var)
EXPOSE 50051

# Use shell form to expand PORT env var (Railway sets this dynamically)
CMD python -m ragzoom.cli server start --host 0.0.0.0 --port ${PORT:-50051} --collect-telemetry
