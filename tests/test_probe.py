import json

import pytest

from probe.sampler import ProbeBudget, deterministic_selection, sampling_rate
from probe.analyzer import ProbeAnalyzer
from probe.review_store import ProbeReviewStore
from probe.store import ProbeStore
from probe.types import ProbeRecord


def record():
    snapshot = {
        "width": 300,
        "height": 250,
        "visible": True,
        "in_viewport": True,
        "fixed": False,
        "child_count": 1,
        "resource_hosts": ["ads.example.test"],
        "connected": True,
    }
    return {
        "probe_id": "probe-1",
        "protocol_version": "ssa-probe-v1",
        "probe_type": "dom_change",
        "domain": "news.test",
        "candidate_fingerprint": "a" * 64,
        "seed_digest": "b" * 16,
        "detector_score": 0.72,
        "detector_route": "jev",
        "original_action": "allow",
        "sampling_rate": 0.005,
        "scheduled_delay_ms": 900,
        "started_at": "2026-10-10T00:00:00+00:00",
        "completed_at": "2026-10-10T00:00:01+00:00",
        "before": snapshot,
        "after": snapshot,
        "delta": {
            "width": 0,
            "height": 0,
            "visibility_changed": False,
            "viewport_changed": False,
            "children": 0,
            "new_resource_hosts": [],
            "mutation_count": 2,
            "disconnected": False,
        },
    }


def test_sampling_rate_is_stratified_and_bounded():
    assert sampling_rate(0.44) == 0
    assert sampling_rate(0.45) == 0.001
    assert sampling_rate(0.60) == 0.005
    assert sampling_rate(0.75) == 0.01
    assert sampling_rate(0.90) == 0


def test_selection_is_reproducible():
    first = deterministic_selection("domain|fingerprint|window", 0.8)
    second = deterministic_selection("domain|fingerprint|window", 0.8)
    assert first == second


def test_budget_enforces_page_and_domain_limits():
    budget = ProbeBudget(page_limit=2, domain_limit=2)
    assert budget.consume("page-a", "news.test", 1)
    assert budget.consume("page-a", "news.test", 1)
    assert not budget.consume("page-a", "news.test", 1)
    assert not budget.consume("page-b", "news.test", 1)
    assert budget.consume("page-b", "news.test", 2)


def test_probe_rejects_unapproved_data_fields():
    value = record()
    value["text"] = "private page text"
    with pytest.raises(ValueError, match="unexpected probe fields"):
        ProbeRecord.from_dict(value)


def test_store_writes_only_validated_probe(tmp_path):
    path = tmp_path / "probes.jsonl"
    store = ProbeStore(path)
    result = store.append(record())
    assert result["accepted"]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "text" not in saved
    assert saved["probe_type"] == "dom_change"
    assert store.stats()["records"] == 1


def test_disabled_store_does_not_persist(tmp_path):
    path = tmp_path / "probes.jsonl"
    result = ProbeStore(path, enabled=False).append(record())
    assert not result["accepted"]
    assert not path.exists()


def test_probe_analyzer_discards_unchanged_observation():
    assessment = ProbeAnalyzer().assess(ProbeRecord.from_dict(record()))
    assert assessment.route == "discard"
    assert assessment.information_value == 0


def test_probe_analyzer_routes_new_ad_host_to_model():
    value = record()
    value["after"] = {**value["after"], "resource_hosts": ["ads.doubleclick.net"]}
    value["delta"] = {**value["delta"], "new_resource_hosts": ["ads.doubleclick.net"]}
    assessment = ProbeAnalyzer().assess(ProbeRecord.from_dict(value))
    assert assessment.route == "model"
    assert "probe_known_ad_host" in assessment.evidence


def test_probe_review_queue_and_resolution_are_separate_from_training(tmp_path):
    pending = tmp_path / "pending.jsonl"
    reviewed = tmp_path / "reviewed.jsonl"
    store = ProbeReviewStore(pending, reviewed)
    probe = ProbeRecord.from_dict(record())
    assessment = ProbeAnalyzer(review_threshold=0).assess(probe)
    queued = store.queue(probe, assessment, None)
    assert queued["queued"]
    assert len(store.list_pending()) == 1
    result = store.resolve(queued["review_id"], 1)
    assert result["saved"]
    assert store.list_pending() == []
    saved = json.loads(reviewed.read_text(encoding="utf-8"))
    assert saved["question"]["label"] == 1
    assert saved["provenance"]["label_source"] == "human_probe_review"
