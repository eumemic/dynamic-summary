# Performance Comparison Report

| Chunk Size | Metric | Baseline | Current | Change | Threshold |
|------------|--------|----------|---------|--------|-----------|
| **400 tokens** | Median error | -171.0 ±80.0 tok | -138.5 ±64.5 tok | ⚪ -32.5 tok (-19.0%)<br>⚪ σ-15.5 (-19%) | ±400.0 tok |
|  | p95 error | +99.0 ±80.0 tok | +162.8 ±64.5 tok | ⚪ +63.8 tok (+64.5%)<br>⚪ σ-15.5 (-19%) | ±400.0 tok |
|  | Within ±10 tokens | 6.45 ±0.00 % | 3.33 ±0.00 % | ⚪ -3.1 pp (-48.3%)<br>⚪ σ+0.0 (±0%) | — |
|  | Avg retries/node | 0.00 ±0.00 | 0.00 ±0.00 | — | — |
|  | Median time/node | 2.67 ±1.02 s | 4.72 ±0.70 s | 🟡 +2.1 s (+76.8%)<br>⚪ σ-0.3 (-31%) | ±5.1 s |
|  | USD per node | $0.0051 ±0.0005 | $0.0066 ±0.0005 | 🟡 +$0.0015 (+30.0%)<br>⚪ σ-0.0001 (-13%) | ±$0.0027 |


✅ No regressions detected
