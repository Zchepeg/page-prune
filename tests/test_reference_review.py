import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review_reference_guided", ROOT / "scripts" / "review_reference_guided.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(identifier, domain):
    return {
        "id": identifier,
        "domain": domain,
        "candidate": {
            "candidate_id": identifier,
            "page": {"url": f"https://{domain}/", "top_level_domain": domain},
            "tag": "DIV",
            "text": "Sponsored",
            "element_id": "ad-slot",
            "classes": ["advertisement"],
            "attributes": {},
            "resource_urls": ["https://ads.example/banner"],
            "width": 300,
            "height": 250,
            "fixed": False,
            "visible": True,
        },
        "suggested_label": "review",
    }


def test_export_reviewed_excludes_insufficient_evidence(tmp_path):
    pool = tmp_path / "pool.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    output = tmp_path / "human.jsonl"
    report = tmp_path / "report.json"
    pool.write_text(
        "\n".join(json.dumps(candidate(str(index), f"site-{index}.test")) for index in range(3)),
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(
            [
                json.dumps({"id": "0", "label": "ad", "annotated_at": "one"}),
                json.dumps({"id": "1", "label": "non_ad", "annotated_at": "two"}),
                json.dumps({"id": "2", "label": "insufficient_evidence", "annotated_at": "three"}),
            ]
        ),
        encoding="utf-8",
    )
    result = MODULE.export_reviewed(pool, annotations, output, report)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["examples"] == 2
    assert [row["question"]["label"] for row in rows] == [0, 1]
    assert all(row["provenance"]["label_source"] == "human_review_reference_guided" for row in rows)
    assert json.loads(report.read_text(encoding="utf-8"))["excluded_insufficient_evidence"] == 1
