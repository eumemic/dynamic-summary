"""High-level wrapper classes for simplified RagZoom usage."""

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import create_store


def _initialize_components(
    index_config: IndexConfig | None,
    query_config: QueryConfig | None,
    operational_config: OperationalConfig | None,
) -> tuple[
    IndexConfig,
    QueryConfig,
    OperationalConfig,
    StorageBackend,
]:
    """Initialize common RagZoom components.

    Note: Retriever and Assembler are no longer pre-initialized as they require
    document-scoped stores. Create them per-request using store.for_document().

    Args:
        index_config: Configuration for document indexing
        query_config: Configuration for querying
        operational_config: Operational configuration (API keys, storage)

    Returns:
        Tuple of initialized components
    """
    index_config = index_config or IndexConfig.load()
    query_config = query_config or QueryConfig()
    operational_config = operational_config or OperationalConfig()

    store = create_store(operational_config)

    return (
        index_config,
        query_config,
        operational_config,
        store,
    )


class RagZoom:
    """High-level synchronous interface to RagZoom.

    Provides a simplified API for common indexing and querying operations.
    """

    def __init__(
        self,
        index_config: IndexConfig | None = None,
        query_config: QueryConfig | None = None,
        operational_config: OperationalConfig | None = None,
    ):
        """Initialize RagZoom with optional configurations.

        Args:
            index_config: Configuration for document indexing
            query_config: Configuration for querying
            operational_config: Operational configuration (API keys, storage)
        """
        (
            self.index_config,
            self.query_config,
            self.operational_config,
            self.store,
        ) = _initialize_components(index_config, query_config, operational_config)

    def index(self, text: str, document_id: str) -> str:
        """Index a document.

        Args:
            text: Document text to index
            document_id: Unique identifier for the document

        Returns:
            Document ID that was indexed
        """
        # jscpd:ignore-start - Legitimate pattern for document-scoped TreeBuilder creation
        # Clear existing data if needed
        self.store.clear_document(document_id)

        # Create document with metadata BEFORE creating TreeBuilder
        content_hash = DocumentStore.compute_content_hash(text)
        self.store.add_document(
            document_id=document_id,
            file_path=None,
            content_hash=content_hash,
            chunk_count=0,  # Will be updated after indexing
            embedding_model=self.index_config.embedding_model,
            summary_model=self.index_config.summary_model,
        )

        # Create document-scoped store and TreeBuilder
        document_store = self.store.for_document(document_id)
        tree_builder = TreeBuilder(
            self.index_config,
            document_store,
            self.operational_config.openai_api_key,
        )

        return tree_builder.add_document(text)
        # jscpd:ignore-end

    # jscpd:ignore-start - Legitimate sync/async pattern duplication
    def query(self, query_text: str, document_id: str) -> str:
        """Query within a specific document.

        Args:
            query_text: Query string
            document_id: Document to search within

        Returns:
            Generated summary response
        """
        # Create document-scoped components
        from openai import OpenAI

        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService

        client = OpenAI(
            api_key=self.operational_config.openai_api_key.get_secret_value()
        )
        document_store = self.store.for_document(document_id)
        embedding_service = EmbeddingService(
            client, document_store, self.query_config.embedding_model
        )
        budget_planner = BudgetPlanner(
            document_store, self.index_config.target_chunk_tokens
        )
        retriever = Retriever(
            self.query_config,
            document_store,
            embedding_service,
            budget_planner,
        )
        assembler = Assembler(document_store)

        # Execute query
        node_scores = retriever.retrieve(query_text, document_id=document_id)
        return assembler.assemble(node_scores)

    # jscpd:ignore-end


class AsyncRagZoom:
    """High-level asynchronous interface to RagZoom.

    Provides a simplified async API for common indexing and querying operations.
    """

    def __init__(  # jscpd:ignore-start
        self,
        index_config: IndexConfig | None = None,
        query_config: QueryConfig | None = None,
        operational_config: OperationalConfig | None = None,
    ):
        """Initialize AsyncRagZoom with optional configurations.

        Args:
            index_config: Configuration for document indexing
            query_config: Configuration for querying
            operational_config: Operational configuration (API keys, storage)
        """  # jscpd:ignore-end
        (
            self.index_config,
            self.query_config,
            self.operational_config,
            self.store,
        ) = _initialize_components(index_config, query_config, operational_config)

    async def index_async(self, text: str, document_id: str) -> str:
        """Index a document asynchronously.

        Args:
            text: Document text to index
            document_id: Unique identifier for the document

        Returns:
            Document ID that was indexed
        """
        # jscpd:ignore-start - Legitimate pattern for document-scoped TreeBuilder creation
        # Clear existing data if needed
        self.store.clear_document(document_id)

        # Create document with metadata BEFORE creating TreeBuilder
        content_hash = DocumentStore.compute_content_hash(text)
        self.store.add_document(
            document_id=document_id,
            file_path=None,
            content_hash=content_hash,
            chunk_count=0,  # Will be updated after indexing
            embedding_model=self.index_config.embedding_model,
            summary_model=self.index_config.summary_model,
        )

        # Create document-scoped store and TreeBuilder
        document_store = self.store.for_document(document_id)
        tree_builder = TreeBuilder(
            self.index_config,
            document_store,
            self.operational_config.openai_api_key,
        )

        return await tree_builder.add_document_async(text)
        # jscpd:ignore-end

    # jscpd:ignore-start - Legitimate sync/async pattern duplication
    async def query_async(self, query_text: str, document_id: str) -> str:
        """Query within a specific document asynchronously.

        Args:
            query_text: Query string
            document_id: Document to search within

        Returns:
            Generated summary response
        """
        # Create document-scoped components
        from openai import OpenAI

        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService

        client = OpenAI(
            api_key=self.operational_config.openai_api_key.get_secret_value()
        )
        document_store = self.store.for_document(document_id)
        embedding_service = EmbeddingService(
            client, document_store, self.query_config.embedding_model
        )
        budget_planner = BudgetPlanner(
            document_store, self.index_config.target_chunk_tokens
        )
        retriever = Retriever(
            self.query_config,
            document_store,
            embedding_service,
            budget_planner,
        )
        assembler = Assembler(document_store)

        # Execute query
        node_scores = await retriever.retrieve_async(
            query_text, document_id=document_id
        )
        return assembler.assemble(node_scores)

    # jscpd:ignore-end
