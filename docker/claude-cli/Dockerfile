FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# The Claude CLI refuses --dangerously-skip-permissions as root,
# so we run as a non-root user.
RUN useradd -m claude
USER claude

# Install the native Claude binary (no Node.js runtime needed).
RUN curl -fsSL https://claude.ai/install.sh | bash

# Make the binary visible to `docker exec` (which doesn't source shell profiles).
ENV PATH="/home/claude/.local/bin:$PATH"

# Persistent container — SDK calls enter via `docker exec -i`
ENTRYPOINT ["sleep", "infinity"]
