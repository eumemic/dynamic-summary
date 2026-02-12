FROM node:22-slim

RUN npm install -g @anthropic-ai/claude-code && \
    npm cache clean --force

# The Claude CLI refuses --dangerously-skip-permissions as root,
# so we run as a non-root user.
RUN useradd -m claude
USER claude

# Persistent container — SDK calls enter via `docker exec -i`
ENTRYPOINT ["sleep", "infinity"]
