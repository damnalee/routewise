# Routing evaluation

60 hand-labelled queries. Savings assume 600 input / 250 output tokens per request at reference prices, 0% cache hits.

| threshold | accuracy | hard recall | easy over-routing | share to large | est. savings vs always-large |
|---|---|---|---|---|---|
| 0.10 | 0.95 | 1.00 | 0.10 | 0.55 | 42% |
| 0.15 | 0.98 | 1.00 | 0.03 | 0.52 | 45% |
| 0.20 | 0.97 | 0.97 | 0.03 | 0.50 | 47% |
| 0.25 | 0.97 | 0.97 | 0.03 | 0.50 | 47% |
| 0.30 | 0.95 | 0.90 | 0.00 | 0.45 | 52% |
| 0.35 | 0.82 | 0.63 | 0.00 | 0.32 | 64% |
| 0.40 | 0.82 | 0.63 | 0.00 | 0.32 | 64% |
| 0.45 | 0.75 | 0.50 | 0.00 | 0.25 | 70% |
| 0.50 | 0.65 | 0.30 | 0.00 | 0.15 | 80% |
| 0.55 | 0.62 | 0.23 | 0.00 | 0.12 | 83% |
| 0.60 | 0.58 | 0.17 | 0.00 | 0.08 | 86% |
