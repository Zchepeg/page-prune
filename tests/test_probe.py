import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateRanker
from jev.backend import BackendResult
from jev.decider import JevDecider
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from probe.sampler import ProbeBudget, deterministic_selection, sampling_rate
from probe.analyzer import ProbeAnalyzer
from probe.review_store import ProbeReviewStore
from probe.store import ProbeStore
from probe.types import ProbeRecord
from service.orchestrator import AdBlockOrchestrator


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


@pytest.mark.parametrize(("width", "height", "evidence", "route"), [
    (0, -40, "layout_contraction", "discard"),
    (45, -40, "layout_resize", "discard"),
    (0, 40, "layout_expansion", "review"),
])
def test_probe_layout_changes_preserve_direction(width, height, evidence, route):
    value = record()
    value["delta"].update(width=width, height=height)
    result = ProbeAnalyzer().assess(ProbeRecord.from_dict(value))
    assert result.evidence == [evidence]
    assert result.route == route


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


def test_probe_preserves_review_evidence_without_starting_or_recovering_model(tmp_path, monkeypatch):
    class Backend:
        ready = False
        probe_calls = 0
        batch_failed = False
        probe_failed = False

        def runtime_status(self):
            return {"ready": self.ready, "phase": "ready" if self.ready else "warming"}

        def decide(self, state, question):
            self.probe_calls += 1
            if self.probe_failed:
                raise RuntimeError("test probe inference error")
            return BackendResult({"ad": 0.1, "non_ad": 0.9}, DecisionLevel.L2, "test", {})

        def decide_many(self, states, question):
            if self.batch_failed:
                raise RuntimeError("test batch failure")
            return [BackendResult({"ad": 0.1, "non_ad": 0.9}, DecisionLevel.L2, "test", {}) for _ in states]

    clock = [100.0]
    monkeypatch.setattr("service.orchestrator.monotonic", lambda: clock[0])
    backend = Backend()
    store = ProbeStore(tmp_path / "probes.jsonl")
    reviews = ProbeReviewStore(tmp_path / "pending.jsonl", tmp_path / "reviewed.jsonl")
    service = AdBlockOrchestrator(CandidateRanker(), JevDecider(backend), BlockingPolicy(),
                                  probe_store=store, probe_reviews=reviews, defer_model_startup=True)

    def observe(index):
        value = record()
        value["probe_id"] = f"probe-{index}"
        value["candidate_fingerprint"] = f"{index:064x}"
        value["delta"]["new_resource_hosts"] = ["ads.doubleclick.net"]
        result = service.record_probe(value)
        assert result["accepted"] and result["review"]["queued"]
        return result

    assert observe(1)["model_status"] == "model_starting"
    assert backend.probe_calls == 0
    backend.ready = True
    backend.probe_failed = True
    assert observe(2)["decision"] is None
    backend.probe_failed = False
    target = Candidate("one", PageContext("https://news.test"), width=300, height=250)
    backend.batch_failed = True
    assert service.process_many([target])[0]["source"] == "model_unavailable"
    assert observe(3)["model_status"] == "model_unavailable"
    clock[0] = 105.0
    assert observe(4)["model_status"] == "model_unavailable"
    assert backend.probe_calls == 1
    backend.batch_failed = False
    assert service.process_many([target])[0]["source"] == "jev"
    assert observe(5)["model_status"] == "decided"
    assert store.stats()["records"] == reviews.stats()["pending"] == 5
    assert sum(bool(item["provenance"]["model_verdict"]) for item in reviews.list_pending()) == 1
    assert service.probe_stats()["outcomes"] == {
        "routes": {"model": 5},
        "model": {"model_starting": 1, "model_unavailable": 3, "decided": 1},
        "review": {"queued": 5},
    }


def test_duplicate_probe_observations_share_one_inference_and_review(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class Backend:
        calls = 0

        def decide(self, state, question):
            self.calls += 1
            started.set()
            assert release.wait(timeout=5)
            return BackendResult({"ad": 0.1, "non_ad": 0.9}, DecisionLevel.L2, "test", {})

    backend = Backend()
    store = ProbeStore(tmp_path / "probes.jsonl")
    reviews = ProbeReviewStore(tmp_path / "pending.jsonl", tmp_path / "reviewed.jsonl")
    service = AdBlockOrchestrator(CandidateRanker(), JevDecider(backend), BlockingPolicy(),
                                  probe_store=store, probe_reviews=reviews)
    value = record()
    value["delta"]["new_resource_hosts"] = ["ads.doubleclick.net"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.record_probe, value)
        try:
            assert started.wait(timeout=5)
            duplicate = pool.submit(service.record_probe, {**value, "probe_id": "probe-2"}).result(timeout=3)
            assert duplicate["model_status"] == "duplicate_review"
            assert duplicate["decision"] is None
            assert not duplicate["review"]["queued"]
        finally:
            release.set()
        result = first.result(timeout=5)

    assert result["model_status"] == "decided" and result["review"]["queued"]
    review_id = result["review"]["review_id"]
    assert duplicate["review"]["review_id"] == review_id
    assert service.record_probe({**value, "probe_id": "probe-3"})["model_status"] == "duplicate_review"
    assert reviews.resolve(review_id, 1)["saved"]
    assert service.record_probe({**value, "probe_id": "probe-4"})["model_status"] == "duplicate_review"
    assert backend.calls == 1
    assert store.stats()["records"] == 4 and reviews.stats() == {"pending": 0, "reviewed": 1}
    outcomes = service.probe_stats()["outcomes"]
    assert outcomes["model"] == {"decided": 1, "duplicate_review": 3}
    assert outcomes["review"] == {"queued": 1, "duplicate": 3}
    assert "news.test" not in json.dumps(outcomes)
