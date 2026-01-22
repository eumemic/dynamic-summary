"""Tests for get_document_is_temporal() repository method."""

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend


def test_get_document_is_temporal_returns_none_for_missing_document(
    sqlite_backend: SQLiteStorageBackend,
) -> None:
    """Return None when document does not exist."""
    result = sqlite_backend.doc_repo.get_document_is_temporal("nonexistent-doc")
    assert result is None


def test_get_document_is_temporal_returns_false_for_non_temporal_document(
    sqlite_backend: SQLiteStorageBackend,
) -> None:
    """Return False for newly created document (default is non-temporal)."""
    sqlite_backend.add_document(
        document_id="doc-non-temporal",
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-nano",
    )
    result = sqlite_backend.doc_repo.get_document_is_temporal("doc-non-temporal")
    assert result is False


def test_get_document_is_temporal_returns_true_for_temporal_document(
    sqlite_backend: SQLiteStorageBackend,
) -> None:
    """Return True for document marked as temporal."""
    sqlite_backend.add_document(
        document_id="doc-temporal",
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-nano",
    )
    # Set the document as temporal using the setter we'll implement next
    sqlite_backend.doc_repo.set_document_is_temporal("doc-temporal", is_temporal=True)

    result = sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal")
    assert result is True
