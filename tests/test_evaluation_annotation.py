import hashlib
import json

from evaluation.annotation import clean_url, freeze_pool, latest_annotations, pool_status


def record(identifier, domain):
    return {
        "id": identifier,
        "domain": domain,
        "candidate": {
            "candidate_id": identifier,
            "page": {"url": f"https://{domain}/", "title": "", "top_level_domain": domain},
            "tag": "DIV",
            "html": "<div data-token='private'>content</div>",
        },
    }


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_clean_url_removes_query_and_fragment():
    assert clean_url("https://example.test/ad?id=secret#part") == "https://example.test/ad"
    assert clean_url("javascript:alert(1)") == ""


def test_binary_freeze_excludes_uncertain_and_assisted_labels(tmp_path):
    pool = [record('a', 'a.test'), record('b', 'b.test'), record('c', 'c.test'), record('d', 'd.test')]
    labels = [{'id': identifier, 'label': label, 'annotation_mode': mode} for identifier, label, mode in [('a', 'ad', 'human_independent'), ('b', 'non_ad', 'human_independent'), ('c', 'insufficient_evidence', 'human_independent'), ('d', 'ad', 'human_confirmed_machine_suggestion')]]
    write_jsonl(tmp_path/'pool.jsonl', pool)
    write_jsonl(tmp_path/'labels.jsonl', labels)
    result = freeze_pool(tmp_path/'pool.jsonl', tmp_path/'labels.jsonl', tmp_path/'output.jsonl', tmp_path/'meta.json', minimum_examples=2, minimum_domains=2, minimum_per_label=1, required_labels={'ad', 'non_ad'}, ground_truth_task='binary_ad_detection', allowed_annotation_modes={'human_independent'})
    assert result['examples'] == 2
    assert result['excluded_annotations'] == 2
    assert {json.loads(line)['id'] for line in (tmp_path/'output.jsonl').read_text(encoding='utf-8').splitlines()} == {'a', 'b'}


def test_review_freeze_preserves_binary_independent_settings(monkeypatch):
    from types import SimpleNamespace
    from scripts import review_evaluation_pool as module

    captured = {}
    def freeze(*args, **kwargs):
        captured.update(kwargs)
        return {'examples': 300}
    monkeypatch.setattr(module, 'freeze_pool', freeze)
    handler = object.__new__(module.Handler)
    handler.path = '/api/freeze'
    handler.authorized = lambda: True
    handler.server = SimpleNamespace(settings={'pool':'pool', 'annotations':'labels', 'output':'out', 'metadata':'meta', 'ground_truth_task':'binary_ad_detection', 'independent_only':True})
    handler.send_json = lambda status, value: captured.update(status=status)
    handler.do_POST()
    assert captured['status'] == 200
    assert captured['required_labels'] == {'ad', 'non_ad'}
    assert captured['allowed_annotation_modes'] == {'human_independent'}


def test_latest_annotations_uses_last_valid_label(tmp_path):
    path = tmp_path / "annotations.jsonl"
    write_jsonl(path, [{"id": "one", "label": "ad"}, {"id": "one", "label": "non_ad"}])
    assert latest_annotations(path)["one"]["label"] == "non_ad"


def test_pool_status_ignores_annotations_outside_pool():
    pool = [record("one", "a.test"), record("two", "b.test")]
    annotations = {"one": {"label": "ad"}, "missing": {"label": "non_ad"}}
    status = pool_status(pool, annotations)
    assert status["reviewed"] == 1
    assert status["remaining"] == 1
    assert status["labels"]["ad"] == 1


def test_freeze_requires_review_thresholds(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    write_jsonl(pool_path, [record("one", "a.test")])
    write_jsonl(annotations_path, [{"id": "one", "label": "ad"}])
    try:
        freeze_pool(pool_path, annotations_path, tmp_path / "out.jsonl", tmp_path / "out.metadata.json")
    except ValueError as exc:
        assert "need_300_reviewed_examples" in str(exc)
    else:
        raise AssertionError("undersized benchmark must not freeze")


def test_freeze_writes_benchmark_and_metadata(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    output = tmp_path / "out.jsonl"
    metadata = tmp_path / "out.metadata.json"
    pool = [record(f"item-{label}-{index}", f"d-{index % 3}.test") for label in ("ad", "non_ad", "insufficient_evidence") for index in range(3)]
    annotations = [{"id": item["id"], "label": item["id"].removeprefix("item-").rsplit("-", 1)[0]} for item in pool]
    write_jsonl(pool_path, pool)
    write_jsonl(annotations_path, annotations)
    result = freeze_pool(pool_path, annotations_path, output, metadata, minimum_examples=9, minimum_domains=3, minimum_per_label=3)
    assert result["examples"] == 9
    assert len(output.read_text(encoding="utf-8").splitlines()) == 9
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["candidate"]["html"] == ""
    assert json.loads(metadata.read_text(encoding="utf-8"))["labels_frozen_before_evaluation"] is True
    assert json.loads(metadata.read_text(encoding="utf-8"))["used_for_supervision"] is False
    assert json.loads(metadata.read_text(encoding="utf-8"))["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_binary_freeze_requires_only_ad_and_non_ad(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    output = tmp_path / "out.jsonl"
    metadata = tmp_path / "out.metadata.json"
    pool = [record(f"item-{label}-{index}", f"d-{index % 3}.test") for label in ("ad", "non_ad") for index in range(3)]
    annotations = [{"id": item["id"], "label": item["id"].removeprefix("item-").rsplit("-", 1)[0]} for item in pool]
    write_jsonl(pool_path, pool)
    write_jsonl(annotations_path, annotations)
    result = freeze_pool(
        pool_path,
        annotations_path,
        output,
        metadata,
        minimum_examples=6,
        minimum_domains=3,
        minimum_per_label=3,
        required_labels={"ad", "non_ad"},
        ground_truth_task="binary_ad_detection",
    )
    assert result["examples"] == 6
    frozen_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    assert frozen_metadata["ground_truth_task"] == "binary_ad_detection"
    assert frozen_metadata["required_labels"] == ["ad", "non_ad"]


def test_freeze_can_exclude_machine_assisted_annotations(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    output = tmp_path / "out.jsonl"
    metadata = tmp_path / "out.metadata.json"
    pool = [record("independent-ad", "a.test"), record("independent-non", "b.test"), record("assisted", "c.test")]
    annotations = [
        {"id": "independent-ad", "label": "ad", "annotation_mode": "human_independent"},
        {"id": "independent-non", "label": "non_ad", "annotation_mode": "human_independent"},
        {"id": "assisted", "label": "non_ad", "annotation_mode": "human_confirmed_machine_suggestion"},
    ]
    write_jsonl(pool_path, pool)
    write_jsonl(annotations_path, annotations)
    result = freeze_pool(
        pool_path,
        annotations_path,
        output,
        metadata,
        minimum_examples=2,
        minimum_domains=2,
        minimum_per_label=1,
        required_labels={"ad", "non_ad"},
        ground_truth_task="binary_ad_detection",
        allowed_annotation_modes={"human_independent"},
    )
    assert result["examples"] == 2
    assert result["excluded_annotations"] == 1
    assert result["annotation_modes"] == ["human_independent"]
