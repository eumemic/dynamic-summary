---
allowed-tools: Task, Read, Grep, Glob
description: Find code areas most needing refactoring for craftsmanship principles
argument-hint: [focus area]
---

# /kaizen
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- System Overview: @README.md
- Architecture: @docs/architecture.md
- Focus Area: "$ARGUMENTS"

## Strategic Guidance

Code quality assessment requires looking beyond syntax to examine fundamental design decisions. The goal is to identify areas where violations of craftsmanship principles create the most technical debt and maintenance burden.

**Assessment Framework**: Evaluate each area against our craftsmanship principles:
- **Clean**: Dead code, unclear purpose, temporary hacks
- **Simple**: Unnecessary complexity, over-engineering, clever solutions
- **Comprehensible**: Poor naming, high cognitive load, hidden dependencies
- **Well-Factored**: Mixed responsibilities, large functions, tight coupling
- **Single Responsibility**: Classes/functions doing multiple things
- **Testable**: Hard-to-test code, side effects, deep dependencies
- **DRY**: Code duplication, missed abstraction opportunities

**Research Methodology**: Dispatch parallel agents to analyze different system areas, then synthesize findings to identify the single most impactful refactoring opportunity.

## Task
Arguments: "$ARGUMENTS"

Execute a comprehensive code quality assessment to identify the area most needing refactoring to align with our craftsmanship principles.

### Phase 1: Parallel Research (Dispatch 10 Agents)

Use the Task tool to launch 10 parallel code-reviewer agents, each analyzing a specific system area:

1. **Core Indexing Algorithm** - TreeBuilder, text splitting, summarization flow
2. **Core Query Algorithm** - Retriever, DP tiling, assembly pipeline  
3. **Storage Layer** - Store, repositories, database abstractions, caching
4. **API & CLI Layer** - REST endpoints, CLI commands, configuration handling
5. **Models & Data Structures** - Node, Document, complexity and responsibilities
6. **Services & Orchestration** - Inter-component communication, coupling
7. **Test Suite Quality** - Test patterns, coverage, maintainability
8. **Error Handling Patterns** - Exception consistency, error propagation
9. **Documentation Accuracy** - Outdated docs, missing information, clarity
10. **Code Duplication** - DRY violations, abstraction opportunities

Each agent should:
- Assess their area against ALL craftsmanship principles
- Identify specific violations with code examples
- Evaluate maintenance burden and technical debt
- Rate severity: Critical (blocks development), High (frequent pain), Medium (occasional friction)

### Phase 2: Synthesis and Recommendation

Based on agent findings:
1. **Identify Patterns**: Common violations across multiple areas
2. **Assess Impact**: Which violations cause the most development friction?
3. **Scope Appropriately**: Find ONE well-bounded refactoring task
4. **Consider Dependencies**: What other areas would benefit from this fix?

### Output Format

**🔍 Research Summary**
[Brief overview of areas analyzed and key findings]

**⚠️ Critical Issues Found**
- [Most severe violations with specific examples]

**🎯 Recommended Refactoring Task**
- **Area**: [Specific module/component]
- **Problem**: [Clear description of violations]
- **Proposed Solution**: [Concrete refactoring approach]
- **Success Criteria**: [How to measure improvement]
- **Estimated Impact**: [Benefits to code quality and development velocity]

**📊 Alternative Candidates**
- [2-3 other high-impact opportunities for future consideration]

## Retrospective
After completing this task, reflect on three levels:
1. **Command Improvement**: How could this command guide future agents better?
2. **Rubric Conformance**: Does this command follow the /command design principles well?
3. **Meta Evolution**: Should the /command rubric itself evolve based on your experience?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.