---
name: kaizen
description: This skill should be used when the user asks to "find refactoring opportunities", "assess code quality", "find technical debt", "what needs cleanup", "code review the codebase", or mentions wanting to improve code craftsmanship.
---

# Kaizen: Code Quality Assessment

Find code areas most needing refactoring for craftsmanship principles.

## Assessment Framework

Evaluate code against craftsmanship principles:

- **Clean**: Dead code, unclear purpose, temporary hacks
- **Simple**: Unnecessary complexity, over-engineering, clever solutions
- **Comprehensible**: Poor naming, high cognitive load, hidden dependencies
- **Well-Factored**: Mixed responsibilities, large functions, tight coupling
- **Single Responsibility**: Classes/functions doing multiple things
- **Testable**: Hard-to-test code, side effects, deep dependencies
- **DRY**: Code duplication, missed abstraction opportunities

## Process

### Phase 1: Parallel Research

Dispatch parallel agents to analyze different system areas:

1. Core algorithms and business logic
2. Data layer and storage abstractions
3. API and interface layer
4. Models and data structures
5. Services and orchestration
6. Test suite quality
7. Error handling patterns
8. Documentation accuracy
9. Code duplication

Each agent should:
- Assess against ALL craftsmanship principles
- Identify specific violations with code examples
- Rate severity: Critical (blocks development), High (frequent pain), Medium (occasional friction)

### Phase 2: Synthesis

Based on findings:
1. **Identify Patterns**: Common violations across areas
2. **Assess Impact**: Which violations cause most friction?
3. **Scope Appropriately**: Find ONE well-bounded refactoring task
4. **Consider Dependencies**: What other areas benefit from this fix?

## Output Format

**Research Summary**
[Brief overview of areas analyzed and key findings]

**Critical Issues Found**
- [Most severe violations with specific examples]

**Recommended Refactoring Task**
- **Area**: [Specific module/component]
- **Problem**: [Clear description of violations]
- **Proposed Solution**: [Concrete refactoring approach]
- **Success Criteria**: [How to measure improvement]
- **Estimated Impact**: [Benefits to code quality]

**Alternative Candidates**
- [2-3 other high-impact opportunities for future consideration]
