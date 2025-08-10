# Performance Comparison Report

| Chunk Size | Metric | Baseline | Current | Change | Threshold |
|------------|--------|----------|---------|--------|-----------|
| **400 tokens** | Median error | -171.0 ±80.0 tok | -120.0 ±73.0 tok | ⚪ -51.0 tok (-29.8%)<br>⚪ σ-7.0 (-9%) | ±400.0 tok |
|  | p95 error | +99.0 ±80.0 tok | +93.5 ±73.0 tok | ⚪ -5.5 tok (-5.6%)<br>⚪ σ-7.0 (-9%) | ±400.0 tok |
|  | Within ±10 tokens | 6.45 ±0.00 % | 0.00 ±0.00 % | ⚪ -6.5 pp (-100.0%)<br>⚪ σ+0.0 (±0%) | — |
|  | Avg retries/node | 0.00 ±0.00 | 0.00 ±0.00 | — | — |
|  | Median time/node | 2.67 ±1.02 s | 5.74 ±1.40 s | 🟡 +3.1 s (+114.8%)<br>⚪ σ+0.4 (+37%) | ±5.1 s |
|  | USD per node | $0.0051 ±0.0005 | $0.0063 ±0.0005 | 🟡 +$0.0012 (+24.3%)<br>⚪ σ-0.0000 (-2%) | ±$0.0027 |


✅ No regressions detected
