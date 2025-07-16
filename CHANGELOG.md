# Changelog

All notable changes to RagZoom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Document Isolation**: Complete namespace separation between indexed documents
  - Queries now require `--document-id` parameter to prevent cross-document contamination
  - Filename is used as default document ID when indexing files
  - New `documents` command to list all indexed documents with metadata
  - `--document-id` parameter added to `clear` command for targeted deletion
  - `--clear` flag added to `index` command for atomic re-indexing
  - API endpoints updated to require `document_id` in query requests
  - Document metadata tracked: file path, index timestamp, chunk count

### Changed
- Query command now requires document ID to be specified with `-d` or `--document-id`
- API `/query` endpoint now requires `document_id` field in request body
- ChromaDB metadata now includes document_id for filtering
- Store methods updated to support document-level operations

### Fixed
- ChromaDB no longer accepts None values in metadata (now uses empty string as default)

## [0.1.0] - Initial Release

### Added
- Hierarchical tree structure with binary organization
- Dynamic resolution that "zooms in" on relevant content
- MMR (Maximal Marginal Relevance) diversity for comprehensive results
- Slope-capped transitions (±1 level) for coherent summaries
- Strict token budget management with configurable limits
- Incremental updates with efficient dirty node tracking
- Node pinning for always-included content
- Sliding queue eviction with freshness decay
- Optional smoothing pass for enhanced coherence
- CLI interface with comprehensive commands
- REST API with FastAPI
- Python API for programmatic access
- Comprehensive test suite with git hooks
- Validation framework for data integrity checks