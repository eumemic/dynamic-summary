# Changelog

All notable changes to RagZoom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **PostgreSQL Migration**: Complete migration from dual-store architecture to single PostgreSQL database
  - **Single Transactional Store**: PostgreSQL with pgvector extension for both tree structure and embeddings
  - **Automatic Docker Management**: Zero-configuration PostgreSQL setup via Docker containers
  - **Full ACID Guarantees**: Atomic operations across all data with rollback capability
  - **Better Performance**: Reduced coordination overhead, comparable or better vector search performance
  - **Enhanced Reliability**: Foreign key constraints prevent orphaned nodes, better error recovery
  - **Simplified Operations**: Single database for backup/restore, unified connection management
- **Document Isolation**: Complete namespace separation between indexed documents
  - Queries now require `--document-id` parameter to prevent cross-document contamination
  - Filename is used as default document ID when indexing files
  - New `documents` command to list all indexed documents with metadata
  - `--document-id` parameter added to `clear` command for targeted deletion
  - `--clear` flag added to `index` command for atomic re-indexing
  - API endpoints updated to require `document_id` in query requests
  - Document metadata tracked: file path, index timestamp, chunk count

### Changed
- **BREAKING**: Replaced SQLite + ChromaDB with PostgreSQL + pgvector (requires re-indexing)
- **BREAKING**: Configuration now uses `RAGZOOM_DATABASE_URL` instead of separate database/ChromaDB paths
- Query command now requires document ID to be specified with `-d` or `--document-id`
- API `/query` endpoint now requires `document_id` field in request body
- Store methods updated to support document-level operations
- Vector search now uses PostgreSQL's native cosine distance instead of ChromaDB

### Removed
- **BREAKING**: ChromaDB dependency and all related configuration
- **BREAKING**: SQLite database support in favor of PostgreSQL-only architecture
- Legacy dual-store coordination code and error handling

### Fixed
- Transactional consistency issues that could leave system in inconsistent state
- Performance bottlenecks from coordinating between two separate databases

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