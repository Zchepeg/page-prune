# Ablation report

Domain-disjoint: `True`
Eligible for performance claims: `True`
Ineligibility reasons: `none`

This eligibility applies only to the historical checkpoint evaluated before the benchmark was reused as development supervision. It does not apply to later checkpoints trained or tuned from the 500-label reference pool.

| Mode | Precision | Recall | F1 | Rule/model coverage | Action FPR | ECE | P95 ms | SSA discoveries | SSA yield |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules_only | 0.9167 | 0.2500 | 0.3929 | 0.7371 | 0.0085 | 0.1089 | 0.037 | 0 | 0.0000 |
| model_only | 0.0755 | 0.0303 | 0.0432 | 0.1097 | 0.0285 | 0.3306 | 110.898 | 0 | 0.0000 |
| rule_model_cascade | 0.8718 | 0.2576 | 0.3977 | 0.7433 | 0.0114 | 0.1931 | 92.733 | 0 | 0.0000 |
| jev_ad_protocol | 0.8718 | 0.2576 | 0.3977 | 0.7433 | 0.0114 | 0.1931 | 92.984 | 0 | 0.0000 |
| jev_ad_protocol_with_ssa | 0.8718 | 0.2576 | 0.3977 | 0.7433 | 0.0114 | 0.1931 | 93.387 | 0 | 0.0000 |

## Contribution against rules-only

| Comparison | Changed predictions | Corrected | Regressed | Recovered ads | Added destructive false positives |
| --- | --- | --- | --- | --- | --- |
| cascade_vs_rules | 3 | 1 | 0 | 1 | 1 |
| protocol_vs_rules | 3 | 1 | 0 | 1 | 1 |
