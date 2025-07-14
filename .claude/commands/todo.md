# /todo

Add a new TODO item to @docs/todos.md based the user's description:

>$ARGUMENTS

Expand the user's brief description into a self-explanatory, information-dense item that provides enough context for a future agent to understand and act on the task.

## Instructions:

1. **Parse the user's description**: Understand the user's request from the description above
2. **Read the current context**: Understand what file, function, or issue the user is working on
3. **Expand the description**: Transform the user's brief note into a comprehensive TODO item that includes:
   - Specific file paths, function names, or component references
   - Current behavior vs expected behavior  
   - Any error messages or symptoms observed
   - Relevant technical details (e.g., framework versions, dependencies)
   - Potential starting points for investigation
3. **Determine priority**: Based on impact and context, assign High/Medium/Low priority
4. **Update @docs/todos.md**: Add the item to the appropriate priority section
5. **Keep it concise**: Be information-dense but avoid unnecessary verbosity

## Examples:

User description: "figure out why this test is intermittently failing"
→ Becomes: "Fix intermittent failure in e2e-tests/tests/claims-buddy/file-upload.spec.ts - 'should handle concurrent uploads' test fails ~30% of runs in CI but passes locally. Likely race condition with file input disable/enable logic. Check ClaimsProcessor.tsx:handleFileUpload state transitions."

User description: "add caching here"  
→ Becomes: "Implement caching for orchestrator/api.py:get_draft endpoint - Currently makes redundant DB queries for draft history. Consider Redis or in-memory LRU cache with 5-minute TTL for draft metadata."

User description: "this is too slow"
→ Becomes: "Optimize vector-store/server.py:search_similar performance - Search queries taking 2-3s for large documents. Profile embedding generation vs pgvector query time. Consider batch processing or async embedding generation."