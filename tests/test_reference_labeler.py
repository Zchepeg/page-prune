import json

import pytest

pytest.importorskip("torch")

from training.reference_labeler import augment_reference_ads, label_candidates, load_reference, train_reference_model


def item(identifier, domain, advertising):
    return {
        "id": identifier,
        "domain": domain,
        "candidate": {
            "candidate_id": identifier,
            "page": {"url": f"https://{domain}/", "top_level_domain": domain},
            "tag": "DIV" if advertising else "ARTICLE",
            "text": "Sponsored" if advertising else "Article body",
            "element_id": "ad-slot" if advertising else "article-content",
            "classes": ["advertisement" if advertising else "content-module"],
            "attributes": {},
            "resource_urls": ["https://doubleclick.net/ads/banner"] if advertising else [],
            "width": 300 if advertising else 760,
            "height": 250 if advertising else 600,
            "fixed": False,
            "visible": True,
        },
    }


def test_reference_loader_joins_all_annotations(tmp_path):
    pool = tmp_path / "pool.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    values = [item("one", "one.test", True), item("two", "two.test", False)]
    pool.write_text("\n".join(json.dumps(value) for value in values), encoding="utf-8")
    annotations.write_text(
        "\n".join(
            [
                json.dumps({"id": "one", "label": "ad", "annotation_mode": "human_independent"}),
                json.dumps({"id": "two", "label": "non_ad", "annotation_mode": "human_confirmed_machine_suggestion"}),
            ]
        ),
        encoding="utf-8",
    )
    joined = load_reference(pool, annotations)
    assert [value["label"] for value in joined] == ["ad", "non_ad"]


def test_reference_loader_uses_latest_annotation(tmp_path):
    pool = tmp_path / "pool.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    pool.write_text(json.dumps(item("one", "one.test", True)), encoding="utf-8")
    annotations.write_text(
        "\n".join(
            [
                json.dumps({"id": "one", "label": "ad"}),
                json.dumps({"id": "one", "label": "non_ad"}),
            ]
        ),
        encoding="utf-8",
    )
    joined = load_reference(pool, annotations)
    assert joined[0]["label"] == "non_ad"


def test_reference_ensemble_labels_clear_new_examples():
    reference = []
    for index in range(120):
        value = item(f"reference-{index}", f"reference-{index % 30}.test", index % 3 == 0)
        value["label"] = "ad" if index % 3 == 0 else "non_ad"
        reference.append(value)
    model = train_reference_model(reference, folds=3, seed=7, precision_target=0.9)
    pool = [item("new-ad", "new-ad.test", True), item("new-content", "new-content.test", False)]
    accepted, review, report = label_candidates(model, pool, max_std=0.5)
    assert not review
    assert [value["question"]["label"] for value in accepted] == [0, 1]
    assert report["reference_examples"] == 120


def test_reference_augmentation_uses_new_domains_without_copying_text():
    reference = [item("reference-ad", "reference.test", True) | {"label": "ad"}]
    records = augment_reference_ads(reference, ["new.test"], ["ads.example"], 3, seed=9)
    assert len(records) == 3
    assert all(record["source"] == "new.test" for record in records)
    assert all(record["question"]["label"] == 0 for record in records)
    assert all("reference.test" not in json.dumps(record) for record in records)
