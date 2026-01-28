# Implementation Plan: Step-Level Chunking

**Spec:** `specs/step-level-chunking.md`
**Status:** IN PROGRESS

## Overview

Replace turn-based transcript chunking with step-based chunking. Currently, messages are grouped into "turns" (user prompt through assistant response cycle), but this creates arbitrarily large leaves that defeat fine-grained temporal queries. With step-level chunking, each meaningful JSONL record becomes its own leaf node with `time_start = time_end = record.timestamp`, enabling precise temporal retrieval.

## Work Items

### Phase 1: Core Dataclasses

- [x] Create `Step` dataclass
  - Spec: specs/step-level-chunking.md § 1. Replace `Turn` with `Step` dataclass
  - Success: `Step` dataclass exists with `uuid: str` and `timestamp: str` fields
  - Test: `test_step_dataclass.py::TestStepDataclass::test_step_has_uuid_field`, `test_step_has_timestamp_field`
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:86-99

- [ ] Update `SyncResult` to use `steps_appended`
  - Spec: specs/step-level-chunking.md § 5. Update `SyncResult`
  - Success: `SyncResult.steps_appended` field replaces `turns_appended`
  - Test: `test_stateless_sync.py::TestExecuteSyncStateless` (after migration)
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:790-808

### Phase 2: New Step-Based Functions

- [ ] Implement `_should_include_record()` helper
  - Spec: specs/step-level-chunking.md § What is a "Step"?
  - Success: Function returns True for user/assistant messages, False for queue-operation/isCompactSummary/isMeta
  - Test: `test_filter_to_steps.py::TestShouldIncludeRecord`
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py (new function near line 331)

- [ ] Implement `filter_to_steps()` function
  - Spec: specs/step-level-chunking.md § 2. Replace `group_into_turns()` with `filter_to_steps()`
  - Success: Function takes `(uuids, records_by_uuid)` and returns `list[Step]` filtering to user/assistant messages
  - Test: `test_filter_to_steps.py::TestFilterToSteps`
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py (new function)

- [ ] Implement `steps_to_append_units()` function
  - Spec: specs/step-level-chunking.md § 3. Replace `turns_to_append_units()` with `steps_to_append_units()`
  - Success: Function creates AppendUnit per step with `time_start = time_end = step.timestamp`
  - Test: `test_step_to_append_unit.py::TestStepsToAppendUnits`
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py (new function)

### Phase 3: Migration - Update Existing Code

- [ ] Simplify `find_truncation_point()` - remove turn boundary detection
  - Spec: specs/step-level-chunking.md § 4. Simplify `find_truncation_point()`
  - Success: Function no longer calls `is_user_message()` for boundary detection; every record is a valid truncation point
  - Test: `test_stateless_sync.py::TestFindTruncationPoint` (update to remove turn boundary tests)
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:193-283

- [ ] Update `execute_sync()` to use step-based functions
  - Spec: specs/step-level-chunking.md § 6. Update `execute_sync()`
  - Success: Uses `filter_to_steps()` and `steps_to_append_units()` instead of `group_into_turns()` and `turns_to_append_units()`
  - Test: `test_stateless_sync.py::TestExecuteSyncStateless`, `test_execute_sync_batch_append.py`
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:859-967 (lines 932-943)

- [ ] Update CLI output messages (turns -> steps)
  - Spec: specs/step-level-chunking.md § 5. Update `SyncResult` (implied)
  - Success: CLI output says "Synced N steps" instead of "Synced N turns"
  - Test: `test_cli.py::TestSyncCommand`
  - Location: integrations/claude-code/src/ragzoom_claude_code/cli.py:65-67, 124-127

### Phase 4: Removal of Dead Code

- [ ] Remove `Turn` dataclass
  - Spec: specs/step-level-chunking.md § Code to Remove
  - Success: `Turn` class no longer exists in codebase
  - Test: Import should fail: `from ragzoom_claude_code.transcript_sync import Turn`
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:67-84

- [ ] Remove `group_into_turns()` function
  - Spec: specs/step-level-chunking.md § Code to Remove
  - Success: Function no longer exists
  - Test: Import should fail: `from ragzoom_claude_code.transcript_sync import group_into_turns`
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:349-404

- [ ] Remove `turns_to_append_units()` function
  - Spec: specs/step-level-chunking.md § Code to Remove
  - Success: Function no longer exists
  - Test: Import should fail: `from ragzoom_claude_code.transcript_sync import turns_to_append_units`
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:492-521

- [ ] Remove `_build_turn()` helper function
  - Spec: specs/step-level-chunking.md § Code to Remove
  - Success: Function no longer exists
  - Test: No references to `_build_turn` in codebase
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:456-489

- [ ] Remove `_is_user_prompt()` helper function
  - Spec: specs/step-level-chunking.md § Code to Remove
  - Success: Function no longer exists
  - Test: No references to `_is_user_prompt` in codebase
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:121-132

- [ ] Remove `is_user_message()` function
  - Spec: specs/step-level-chunking.md § Code to Remove
  - Success: Function no longer exists (verify not used elsewhere first)
  - Test: Import should fail, no other usages
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:135-165

- [ ] Remove `_is_command_output_or_expansion()` helper function
  - Spec: specs/step-level-chunking.md § Code to Remove
  - Success: Function no longer exists
  - Test: No references to `_is_command_output_or_expansion` in codebase
  - Location: integrations/claude-code/src/ragzoom_claude_code/transcript_sync.py:86-118

### Phase 5: Test Updates

- [ ] Replace `test_group_into_turns.py` with `test_filter_to_steps.py`
  - Spec: specs/step-level-chunking.md § Test Updates
  - Success: New test file covers all step filtering cases: user/assistant included, toolUseResult included as own step, meta filtered, compaction filtered, queue-operation filtered, records without timestamps skipped
  - Test: `pytest integrations/claude-code/tests/test_filter_to_steps.py -v`
  - Location: integrations/claude-code/tests/test_group_into_turns.py (delete), integrations/claude-code/tests/test_filter_to_steps.py (create)

- [ ] Replace `test_turn_dataclass.py` with `test_step_dataclass.py`
  - Spec: specs/step-level-chunking.md § Test Updates
  - Success: Tests for `Step` dataclass covering uuid, timestamp fields
  - Test: `pytest integrations/claude-code/tests/test_step_dataclass.py -v`
  - Location: integrations/claude-code/tests/test_turn_dataclass.py (delete), integrations/claude-code/tests/test_step_dataclass.py (create)

- [ ] Replace `test_turn_to_append_unit.py` with `test_step_to_append_unit.py`
  - Spec: specs/step-level-chunking.md § Test Updates
  - Success: Tests for `steps_to_append_units()` verifying `time_start = time_end` per step
  - Test: `pytest integrations/claude-code/tests/test_step_to_append_unit.py -v`
  - Location: integrations/claude-code/tests/test_turn_to_append_unit.py (delete), integrations/claude-code/tests/test_step_to_append_unit.py (create)

- [ ] Update `test_stateless_sync.py`
  - Spec: specs/step-level-chunking.md § Test Updates
  - Success: Remove `TestIsUserMessage` class, update `TestFindTruncationPoint` to not test turn boundaries
  - Test: `pytest integrations/claude-code/tests/test_stateless_sync.py -v`
  - Location: integrations/claude-code/tests/test_stateless_sync.py:20-75 (remove), :77-388 (update)

- [ ] Update test files referencing `turns_appended` to use `steps_appended`
  - Spec: specs/step-level-chunking.md § 5. Update `SyncResult`
  - Success: All 9 test files updated to use `steps_appended`
  - Test: `grep -r "turns_appended" integrations/claude-code/tests/` returns empty
  - Location:
    - integrations/claude-code/tests/test_cli.py
    - integrations/claude-code/tests/test_revert_turn_granularity.py
    - integrations/claude-code/tests/test_stateless_sync_integration.py
    - integrations/claude-code/tests/test_synced_document_temporal_sqlite.py
    - integrations/claude-code/tests/test_transcript_formatting.py
    - integrations/claude-code/tests/test_transcript_summarization_guidance.py
    - integrations/claude-code/tests/test_transcript_temporal_queries_sqlite.py
    - integrations/claude-code/tests/test_turn_level_tracking.py
    - integrations/claude-code/tests/test_unified_identity.py

### Phase 6: Documentation Updates

- [ ] Update `integrations/claude-code/CLAUDE.md` - Turn-Level Granularity section
  - Spec: specs/step-level-chunking.md (implied)
  - Success: Documentation refers to "step-level" granularity instead of "turn-level"
  - Test: Manual review
  - Location: integrations/claude-code/CLAUDE.md:15-22

## Verification

After all phases complete:
1. `pytest integrations/claude-code/tests/ -v` - all tests pass
2. `grep -r "Turn\|group_into_turns\|turns_to_append_units\|turns_appended" integrations/claude-code/src/` - no matches (except imports in `__init__.py` if any)
3. `dmypy check integrations/claude-code/src/` - type checking passes
