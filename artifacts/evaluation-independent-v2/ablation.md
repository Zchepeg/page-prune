# Ablation report

Domain-disjoint: `True`
Eligible for performance claims: `True`
Ineligibility reasons: `none`

| Mode | Precision | Recall | F1 | Rule/model coverage | Action FPR | ECE | P95 ms | SSA discoveries | SSA yield |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules_only | 0.9643 | 0.3649 | 0.5294 | 0.7776 | 0.0024 | 0.0675 | 0.036 | 0 | 0.0000 |
| model_only | 0.0000 | 0.0000 | 0.0000 | 0.9918 | 0.0000 | 0.0907 | 109.878 | 0 | 0.0000 |
| rule_model_cascade | 0.9032 | 0.3784 | 0.5333 | 0.9878 | 0.0024 | 0.0630 | 92.787 | 0 | 0.0000 |
| jev_ad_protocol | 0.9032 | 0.3784 | 0.5333 | 0.9878 | 0.0024 | 0.0631 | 92.845 | 0 | 0.0000 |
| jev_ad_protocol_with_ssa | 0.9032 | 0.3784 | 0.5333 | 0.9878 | 0.0024 | 0.0631 | 92.598 | 0 | 0.0000 |

## Contribution against rules-only

| Comparison | Changed predictions | Corrected | Regressed | Recovered ads | Added destructive false positives |
| --- | --- | --- | --- | --- | --- |
| cascade_vs_rules | 103 | 68 | 0 | 1 | 0 |
| protocol_vs_rules | 103 | 68 | 0 | 1 | 0 |
