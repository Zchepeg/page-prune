# Ablation report

Domain-disjoint: `True`
Eligible for performance claims: `False`
Ineligibility reasons: `independent_human_annotation_required, fewer_than_300_examples, fewer_than_30_domains, fewer_than_50_examples_in_a_label`

| Mode | Precision | Recall | F1 | Rule/model coverage | Action FPR | ECE | P95 ms | SSA discoveries | SSA yield |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules_only | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0000 | 0.0833 | 0.114 | 0 | 0.0000 |
| model_only | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.5716 | 75.037 | 0 | 0.0000 |
| rule_model_cascade | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0000 | 0.0210 | 68.231 | 0 | 0.0000 |
| jev_ad_protocol | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0000 | 0.0210 | 68.132 | 0 | 0.0000 |
| jev_ad_protocol_with_ssa | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0000 | 0.0210 | 68.117 | 0 | 0.0000 |
