# Implementation Summary: Issue #11 - Remove dead dirty node marking code

## Overview
Successfully removed all dead code related to the dirty node marking feature that was never used in production.

## Changes Made

### 1. Database Schema (`ragzoom/store.py`)
- Removed `is_dirty` field from `TreeNode` model
- Updated SQL migration to remove `is_dirty` column

### 2. Store Methods (`ragzoom/store.py`)
- Removed `mark_dirty_upward()` method - was never called in production
- Removed `get_dirty_nodes()` method
- Updated `update_summary()` to remove `is_dirty` flag clearing

### 3. Retrieval Logic (`ragzoom/retrieve.py`)
- Removed `_refresh_dirty_nodes_async()` method
- Removed call to refresh dirty nodes from `retrieve_async()`
- Removed `_refreshed_node_ids` tracking attribute

### 4. Tree Builder (`ragzoom/index.py`)
- Removed `refresh_nodes_async()` method

### 5. API Endpoints (`ragzoom/api.py`)
- Removed `/recompute` endpoint

### 6. Configuration (`ragzoom/config.py`)
- Removed `dirty_refresh_limit` configuration field

### 7. Test Utilities (`tests/mock_store.py`)
- Removed `dirty_nodes` set tracking
- Removed `mark_dirty_upward()` method
- Removed `get_dirty_nodes()` method
- Updated `update_summary()` to remove dirty flag handling
- Removed `is_dirty` filtering from mock queries

### 8. Documentation Updates
- Updated `README.md` to remove reference to dirty node tracking
- Updated `CHANGELOG.md` to remove reference to dirty node tracking
- Updated `docs/api-reference.md` to remove `/recompute` endpoint and `dirty_refresh_limit` config
- Updated `docs/architecture.md` to remove reference to async dirty node refresh
- Updated `docs/developer-guide.md` to remove dirty nodes from state tracking
- Added notes to archive documentation indicating the feature is not implemented

## Impact
- No production functionality affected since `mark_dirty_upward()` was never called
- Simplified codebase by removing ~200 lines of dead code
- Reduced complexity in retrieval flow
- Cleaner API surface without unused endpoints

## Testing
The dirty node functionality was only used in tests. Existing tests that relied on this functionality will need to be updated or removed separately.