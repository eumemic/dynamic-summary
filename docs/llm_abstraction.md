LLM Abstraction (Protocols)

- EmbeddingModel: `embed(texts: Sequence[str]) -> list[list[float]]`
- ChatModel: `complete(messages, temperature?, max_tokens?, reasoning_effort?) -> {content, usage}`

Business logic

- EmbeddingBatcher orchestrates validation + batch splitting over `EmbeddingModel`.
- Summarizer handles prompt shaping, token budgeting, retries, telemetry over `ChatModel`.

Adapters

- OpenAIEmbeddingModel and OpenAIChatModel implement these protocols (lazy SDK import).
- Tests can use MockEmbeddingModel and MockChatModel (`ragzoom/testing/mocks.py`).

Migration notes

- `LLMService` now composes the batcher and summarizer. Existing code can keep using it.
- `IndexerRuntime` constructs `LLMService` by default; can inject a custom one.
- Tests use `IndexerRuntimeHarness` which provides mock implementations via protocol injection.

