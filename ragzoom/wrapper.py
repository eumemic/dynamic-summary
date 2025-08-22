"""High-level wrapper classes for simplified RagZoom usage."""

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import Store


def _initialize_components(
    index_config: IndexConfig | None,
    query_config: QueryConfig | None,
    operational_config: OperationalConfig | None,
) -> tuple[
    IndexConfig,
    QueryConfig,
    OperationalConfig,
    Store,
    TreeBuilder,
    Retriever,
    Assembler,
]:
    """Initialize common RagZoom components.

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

    store = Store(operational_config)
    tree_builder = TreeBuilder(index_config, store, operational_config.openai_api_key)
    retriever = Retriever(query_config, store, operational_config.openai_api_key)
    assembler = Assembler(store)

    return (
        index_config,
        query_config,
        operational_config,
        store,
        tree_builder,
        retriever,
        assembler,
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
            self.tree_builder,
            self.retriever,
            self.assembler,
        ) = _initialize_components(index_config, query_config, operational_config)

    def index(self, text: str, document_id: str) -> str:
        """Index a document.

        Args:
            text: Document text to index
            document_id: Unique identifier for the document

        Returns:
            Document ID that was indexed
        """
        return self.tree_builder.add_document(text, document_id=document_id)

    def query(self, query_text: str, document_id: str) -> str:
        """Query within a specific document.

        Args:
            query_text: Query string
            document_id: Document to search within

        Returns:
            Generated summary response
        """
        node_scores = self.retriever.retrieve(query_text, document_id=document_id)
        return self.assembler.assemble(node_scores)


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
            self.tree_builder,
            self.retriever,
            self.assembler,
        ) = _initialize_components(index_config, query_config, operational_config)

    async def index_async(self, text: str, document_id: str) -> str:
        """Index a document asynchronously.

        Args:
            text: Document text to index
            document_id: Unique identifier for the document

        Returns:
            Document ID that was indexed
        """
        return await self.tree_builder.add_document_async(text, document_id=document_id)

    async def query_async(self, query_text: str, document_id: str) -> str:
        """Query within a specific document asynchronously.

        Args:
            query_text: Query string
            document_id: Document to search within

        Returns:
            Generated summary response
        """
        node_scores = await self.retriever.retrieve_async(
            query_text, document_id=document_id
        )
        return self.assembler.assemble(node_scores)
