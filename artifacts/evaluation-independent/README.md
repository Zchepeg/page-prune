# Independent evaluation summary

This directory contains aggregate results only. The frozen candidates, domains, annotations, screenshots, and per-example predictions are intentionally excluded from the repository.

- Evaluation date: 2026-10-22
- Task: binary ad detection
- Independent human annotations: 483
- Domains: 55
- Advertising examples: 132
- Non-advertising examples: 351
- Machine-assisted annotations excluded: 17
- Training-domain overlap: 0
- Frozen benchmark SHA-256: `7aeae79e0a779169ce92814d773bdd7612a1ab2f4263b267977659eb75832f0d`

The current Qwen checkpoint does not outperform the rules-only baseline. These results describe this checkpoint and configuration only; they are not a claim of general production accuracy.

Compared with rules-only, the protocol changed 3 of 483 predictions, recovered one additional ad, introduced one additional destructive false-positive action, and added about 93 ms of P95 latency. The benchmark later became development supervision for the reference-guided data pipeline, so it must not be reused to claim performance for a newly trained checkpoint.
