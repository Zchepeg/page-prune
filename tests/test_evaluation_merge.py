import json

from scripts.merge_evaluation_pools import merge_annotations, merge_pools


def record(identifier, domain, element_id):
    return {
        "id": identifier,
        "domain": domain,
        "candidate": {
            "candidate_id": identifier,
            "page": {"url": f"https://{domain}/", "top_level_domain": domain},
            "tag": "DIV",
            "text": "",
            "html": "",
            "element_id": element_id,
            "classes": [],
            "attributes": {},
            "resource_urls": [],
            "width": 300,
            "height": 250,
            "fixed": False,
            "visible": True,
        },
        "review_context": {"screenshot": ""},
    }


def test_merge_pools_deduplicates_and_excludes_domains(tmp_path):
    first = tmp_path / "first" / "pool.jsonl"
    second = tmp_path / "second" / "pool.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    shared = record("one", "fresh.example", "slot")
    first.write_text(json.dumps(shared) + "\n", encoding="utf-8")
    second.write_text(
        json.dumps({**shared, "id": "duplicate"}) + "\n" + json.dumps(record("blocked", "used.example", "other")) + "\n",
        encoding="utf-8",
    )

    report = merge_pools([first, second], tmp_path / "merged" / "pool.jsonl", {"used.example"})

    assert report["examples"] == 1
    assert report["domains"] == 1
    assert report["rejected"] == {"duplicate_candidate": 1, "excluded_domain": 1}


def test_merge_annotations_keeps_latest_valid_annotation_for_merged_pool(tmp_path):
    pool = tmp_path / "pool.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "annotations.jsonl"
    pool.write_text(json.dumps(record("one", "fresh.example", "slot")) + "\n", encoding="utf-8")
    first.write_text(json.dumps({"id": "one", "label": "non_ad", "annotation_mode": "human_independent"}) + "\n", encoding="utf-8")
    second.write_text(
        json.dumps({"id": "one", "label": "ad", "annotation_mode": "human_independent"})
        + "\n"
        + json.dumps({"id": "missing", "label": "ad", "annotation_mode": "human_independent"})
        + "\n",
        encoding="utf-8",
    )

    report = merge_annotations([first, second], pool, output)

    assert report["annotations"] == 1
    assert report["labels"] == {"ad": 1}
    assert report["rejected_annotations"] == {"unknown_candidate": 1}
