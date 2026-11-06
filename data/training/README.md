# Training data

Use JSONL and split by website/domain, not by individual DOM node. Nodes from the same
site must not appear in both training and calibration/test partitions.

Each line has one semantic decision. New checkpoints learn `ad` and `non_ad`.
Low-confidence or conflicting predictions are rejected as `insufficient_evidence` by
the decision protocol. Legacy label `2` records are retained for provenance but are
excluded from binary-selective training.

The normalized model input follows `data/schema/model_state_schema.json`. Legacy flat
and nested records are converted to this contract when loaded.

```json
{"id":"site-001","source":"example.test","state":{"schema_version":"candidate-features-v2","detector_evidence":["ad_semantic_marker"],"tag":"IFRAME","element_id":"ad-slot","classes":["advertisement"],"attributes":{},"resource_urls":["https://ads.example/banner"],"geometry":[300,250],"fixed":false,"visible":true,"page_domain":"example.test","text":"Sponsored"},"question":{"instruction":"Classify this web element.","options":["ad","non_ad"],"label":0}}
```

Do not put cookies, authentication data, form values, or complete page HTML in this data.

Browser candidates selected for review remain unlabeled in `pending_review.jsonl`. While the
local service is running, open `http://127.0.0.1:8765/review` to mark them as `ad`, `non_ad`,
or `insufficient_evidence`. Confirmed records are appended to `reviewed.jsonl`; only confirmed
binary labels are eligible for supervised training.

Local reference-guided training and review files are excluded from Git. Their content-free
statistics, provenance boundary, and SHA-256 values are published in
`reference_guided_v1_manifest.json`; the construction procedure is documented in
`docs/reference-guided-data.md`.
