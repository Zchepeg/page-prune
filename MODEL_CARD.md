# JEV Ad Blocker Qwen Decision Model

## Model summary

The decision model uses Qwen3-4B-Base with a LoRA adapter and a pointer-style classification head. The current local checkpoint uses binary `ad` and `non_ad` training with confidence and margin rejection; `insufficient_evidence` remains a protocol outcome rather than a learned content class. Blocking, hiding, allowing, and review remain policy decisions outside the model. The default local runtime loads the backbone with NF4 4-bit double quantization and keeps training outside the decision service.

Runtime inference is governed by `JEV-Ad Decision Protocol v1`. High-confidence deterministic rules bypass the model, ambiguous candidates use the model, and near-boundary rule decisions enter a challenge route. Rule-model conflicts are downgraded to reviewable uncertainty instead of triggering an irreversible block. The complete contract is documented in `docs/decision-protocol.md`.

SSA probe records use the separate `probe-ad-classification-v1` question. This path is experimental and out of the original candidate-state training distribution. Its output is review metadata only: it cannot block a page, create a training label, or promote a model without human review and independent evaluation.

Model weights are not included in the source repository. The local development checkpoint is stored under `runs/`, which is excluded from Git.

## Intended use

- Classify ambiguous advertising candidates after deterministic browser and URL rules.
- Run locally on a user-controlled machine.
- Support research, evaluation, and human-reviewed data collection.

The model is not intended to inspect complete page HTML, replace deterministic filter lists, or make security decisions unrelated to advertising.

## Inputs and privacy

Inputs contain bounded, sanitized element state such as tag name, short text, selected attributes, dimensions, page domain, and resource URLs without credentials or query secrets. Cookies, authorization headers, form values, and complete page HTML are excluded.

## Training data

The current training pipeline combines generated bootstrap records, curated tracker records, deterministic-rule captures, human-reviewed samples, and a reference-guided weak-label set. The latter uses 500 human annotations to fit a domain-folded structural ensemble, accepts only low-disagreement predictions beyond out-of-fold thresholds, and keeps uncertain candidates outside training. Class-balancing counterfactual records are identified separately in provenance. All records are normalized to the versioned `candidate-features-v2` contract before tokenization. Binary training excludes synthetic abstention labels, creates a domain-disjoint validation split, applies early stopping, and restores the best validation checkpoint before calibration. Data provenance is stored with each record. Source references and licensing boundaries are documented in `REFERENCES.md` and `THIRD_PARTY_NOTICES.md`.

## Current evaluation status

The configured binary checkpoint was evaluated on a frozen benchmark containing 490 independently human-labelled browser candidates from 63 domains: 74 ads and 416 non-ads. There is no domain overlap with the 401 training domains, labels were frozen before evaluation, and the benchmark has not been used for supervision. The pool combines broad site sampling with a second ad-enriched collection pass, so prevalence-dependent metrics must not be interpreted as natural-web prevalence. Raw candidates and annotations remain local; the de-identified aggregate report is stored in `artifacts/evaluation-independent-v2/`.

| Mode | Precision | Recall | F1 | False-positive rate | Accuracy | Coverage | ECE | P95 latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rules only | 0.9643 | 0.3649 | 0.5294 | 0.0024 | 0.9020 | 0.7776 | 0.0675 | 0.036 ms |
| Model only | 0.0000 | 0.0000 | 0.0000 | 0.0072 | 0.8429 | 0.9918 | 0.0907 | 109.878 ms |
| Rule-model cascade | 0.9032 | 0.3784 | 0.5333 | 0.0072 | 0.9000 | 0.9878 | 0.0630 | 92.787 ms |
| JEV-Ad protocol | 0.9032 | 0.3784 | 0.5333 | 0.0072 | 0.9000 | 0.9878 | 0.0631 | 92.845 ms |

The cascade changed 103 predictions, corrected 68 rule abstentions, recovered one additional ad, and added no destructive false-positive actions. Recall increased by 1.35 percentage points and F1 by 0.39 percentage points, while classification precision fell by 6.11 percentage points and P95 latency increased by about 92.8 ms. The model-only mode detected no true ads and produced three false ad classifications. The checkpoint therefore remains suitable only as a fail-open gray-zone research component; it is not suitable as a standalone blocker, and its measured cascade benefit is too small to justify a strong model-effectiveness claim.

The previous 483-example report in `artifacts/evaluation-independent/` remains a historical baseline only because its annotation pool was later used as development supervision. Generated calibration and smoke results remain useful only for pipeline verification and are not evidence of real-world accuracy.

## New frozen candidate evaluation, 2026-11-01

Five previously frozen candidates were tested after all 352 independent human labels were frozen (69 ads, 283 non-ads, 36 new domains). The rules baseline acted on 35 ads with no non-ad actions; the TF-IDF cascade acted on 51 with no non-ad actions. Qwen v2 seeds 42/7 acted on 56/62 ads with 1/3 non-ad actions; Qwen v3 seeds 42/7 acted on 66/54 ads with 5/2 non-ad actions. Per-comparison paired domain intervals support additional action recall on this pool, while false-positive and compute costs remain material. No best seed was selected or deployed; these candidates are distinct from the configured historical checkpoint. See `docs/fresh-blind-interpretation.md` for uncertainty, timing boundaries and native-input details. The mixed candidate sample does not estimate natural-web blocking rates, and this test must be retired for future versions if it influences model selection or tuning.

Retrospective paired analysis of the historical v2 predictions used 10,000 domain-cluster bootstrap draws over 63 domains. The cascade-minus-rules recall interval was [0, 0.047059] and the classification F1 interval was [-0.024506, 0.046276]. The ad-recall exact McNemar test gave p=1.0 (one improvement, no regressions); it assumes independent candidate pairs and is supplementary to the cluster analysis. This does not establish a significant ad-recall benefit. Details are in `docs/historical-paired-statistics.md`; these are not new candidate-model results.

## Limitations

- Generated and rule-derived examples may contain shortcuts that do not generalize to unseen websites.
- Language, region, layout, anti-adblock behavior, and first-party advertising can shift over time.
- Incorrect blocking can break page navigation or hide legitimate content.
- A local 4B model has nontrivial cold-start latency and quantization can change predictions near class boundaries.
- The current model-only result has zero ad recall on the independent v2 benchmark.
- The current cascade improves F1 only slightly and adds substantial inference latency.
- The v2 benchmark includes an ad-enriched sampling pass and must not be used to estimate natural-web ad prevalence.
- Once v2 influences training, rules, prompts, thresholds, or model selection, it is no longer a valid blind test for the resulting version.

## Deployment safeguards

- Deterministic high-confidence cases bypass the model.
- Ambiguous states are deduplicated, processed in memory-bounded micro-batches, and capped per browser request.
- Candidates deferred by the model budget fail open and are not cached as semantic decisions.
- Protected semantic elements receive stricter policy treatment.
- Uncertain results can be routed to review.
- Service or model failure defaults to fail-open behavior.
- Candidate checkpoints are not deployed automatically after training.
- The decision service cannot start a training process under the default configuration.
- The configured CUDA allocator budget is 4096 MB; the measured local smoke run peaked at 3836.1 MB allocated and 4084.0 MB reserved.

## Licensing

Project-owned model integration code is covered by the repository Apache-2.0 license. Base-model weights, adapters, datasets, compiled rules, and third-party materials may have separate terms. Verify and document those terms before distributing model weights.
