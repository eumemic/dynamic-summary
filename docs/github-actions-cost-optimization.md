# GitHub Actions Cost Optimization Guide

This document outlines the optimizations made to reduce GitHub Actions costs and provides guidance for future improvements.

## Cost Reduction Summary

The following optimizations have been implemented to reduce CI/CD costs:

### 1. Combined Static Analysis Jobs
- **Before**: 4 separate jobs (lint, format, typecheck, security)
- **After**: 1 combined `static-analysis` job
- **Savings**: ~75% reduction in runner initialization overhead for these checks
- **Impact**: Saves ~3-4 minutes of billable time per PR

### 2. Optimized Test Parallelization
- **Before**: `-n 8` (8 parallel processes on 2-core runners)
- **After**: `-n 2` (matching actual core count)
- **Impact**: More efficient CPU usage, potentially faster overall execution

### 3. Smart Workflow Triggers
- **Path filters**: Skip CI when only docs/configs change
- **Draft PR detection**: Skip expensive tests on draft PRs
- **Coverage only on master**: Run coverage reports only on master pushes
- **Python matrix conditionally**: Skip multi-version tests on drafts

### 4. Enhanced Caching
- **MyPy cache**: Cache type checking results
- **Pytest cache**: Cache test discovery and results
- **Separate cache keys**: Different caches for different job types

## Estimated Cost Savings

Based on these optimizations:
- **Per PR**: ~30-50% reduction in compute minutes
- **Draft PRs**: ~70% reduction (only fast tests + static analysis)
- **Doc-only PRs**: ~95% reduction (workflows skip entirely)

## Additional Cost-Saving Strategies

### Short Term (Already Implemented)
1. ✅ Combine lightweight jobs
2. ✅ Add path filters
3. ✅ Optimize parallelization
4. ✅ Skip expensive tests on drafts
5. ✅ Better caching

### Medium Term (Consider Next)
1. **Self-hosted runners** for expensive jobs:
   ```yaml
   runs-on: self-hosted
   ```
   - Best for: Performance benchmarks, integration tests
   - Setup: Use a spare machine or cloud instance

2. **Merge queues** to batch PR tests:
   - Enable GitHub merge queues
   - Multiple PRs share CI runs

3. **Fail-fast strategy**:
   ```yaml
   strategy:
     fail-fast: true
   ```
   - Stop all matrix jobs if one fails

### Long Term Considerations

1. **Alternative CI platforms**:
   - CircleCI: 3,000 free minutes vs GitHub's 2,000
   - Self-hosted Jenkins: Higher maintenance, zero per-minute cost
   - Hybrid approach: Use GitHub Actions for simple checks, alternatives for heavy lifting

2. **Workflow optimization**:
   - Use `workflow_run` to chain dependent workflows
   - Implement manual approval gates for expensive operations
   - Use repository dispatch for selective triggering

## Monitoring Usage

To track your GitHub Actions usage:
1. Go to Settings → Billing → Actions
2. Monitor daily usage patterns
3. Set up billing alerts at 50%, 75%, 90% thresholds

## Best Practices

1. **Always use concurrency groups** to cancel outdated runs
2. **Cache aggressively** but with proper cache keys
3. **Use matrix strategies sparingly** - they multiply costs
4. **Optimize Docker builds** with layer caching
5. **Review workflow runs** regularly to identify inefficiencies

## Emergency Cost Controls

If approaching budget limits:
1. Disable non-critical workflows temporarily
2. Require manual approval for workflow runs
3. Limit concurrent workflow runs
4. Consider switching to nightly builds instead of per-PR

Remember: The goal is to maintain code quality while managing costs effectively.