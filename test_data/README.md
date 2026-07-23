# Test Data for RagZoom

This directory contains sample documents for testing the RagZoom system:

## Files

1. **moby_dick.txt** (1.2 MB, ~22K lines)
   - Classic novel by Herman Melville
   - Good for testing narrative text processing
   - Downloaded from Project Gutenberg
   - **moby_dick_ci.txt** (~514KB): line-aligned prefix slice used by the CI benchmark workflows (size-matched to the former CI corpus)

2. **bible_kjv.txt** (4.4 MB, ~100K lines)
   - King James Version of the Bible
   - Tests handling of repetitive terms and structured text
   - Downloaded from Project Gutenberg
   - **moby_dick_ci.txt** (~514KB): line-aligned prefix slice used by the CI benchmark workflows (size-matched to the former CI corpus)

3. **sample_chat_log.txt** (2.8 KB, 54 lines)
   - Synthetic 100-turn chat conversation
   - Tests conversational/dialogue text
   - Created to simulate team standup meeting

## Usage

```bash
# Index Moby Dick
ragzoom index test_data/moby_dick.txt --document-id moby-dick

# Query examples
ragzoom query "What happens to Ahab?"
ragzoom query "Tell me about the white whale"

# Index Bible subset (first 1000 lines for testing)
head -n 1000 test_data/bible_kjv.txt | ragzoom index - --document-id bible-subset

# Index chat log
ragzoom index test_data/sample_chat_log.txt --document-id chat-log
ragzoom query "What performance issues were discussed?"
```