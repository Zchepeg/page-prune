import json

from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateAssessment
from feedback.training_capture import TrainingCaptureStore
from jev.types import Decision, DecisionLevel, Verdict


def ad_candidate() -> Candidate:
    return Candidate(
        candidate_id="ad-1",
        page=PageContext(
            "https://news.test/article?token=secret",
            top_level_domain="news.test",
        ),
        tag="IFRAME",
        text="Sponsored placement",
        element_id="top-ad",
        classes=["advertisement"],
        attributes={"aria_label": "advertisement"},
        resource_urls=["https://user:pass@ads.doubleclick.net/banner?uid=123#track"],
        width=728,
        height=90,
        visible=True,
    )


def test_strong_rule_ad_is_saved_without_url_secrets(tmp_path):
    path = tmp_path / "auto_ads.jsonl"
    store = TrainingCaptureStore(path)
    assessment = CandidateAssessment(1.0, ["known_ad_host", "common_ad_dimensions"], "block")
    result = store.capture(ad_candidate(), assessment)
    assert result.saved
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["question"]["label"] == 0
    assert record["provenance"]["label_source"] == "deterministic_rule"
    assert record["state"]["resource_urls"] == ["https://ads.doubleclick.net/banner"]
    assert "token" not in json.dumps(record["state"])


def test_capture_deduplicates_and_enforces_domain_limit(tmp_path):
    store = TrainingCaptureStore(tmp_path / "auto_ads.jsonl", per_domain_limit=1)
    assessment = CandidateAssessment(1.0, ["known_ad_host"], "block")
    assert store.capture(ad_candidate(), assessment).saved
    assert store.capture(ad_candidate(), assessment).reason == "duplicate"
    changed = ad_candidate()
    changed.element_id = "second-ad"
    assert store.capture(changed, assessment).reason == "domain_limit"


def test_model_or_ambiguous_evidence_cannot_self_label(tmp_path):
    review_path = tmp_path / "pending_review.jsonl"
    store = TrainingCaptureStore(tmp_path / "auto_ads.jsonl", review_path=review_path)
    assessment = CandidateAssessment(0.8, ["common_ad_dimensions"], "jev")
    result = store.capture(ad_candidate(), assessment)
    assert not result.saved
    assert not result.queued
    assert result.reason == "below_review_threshold"


def test_two_independent_rule_signals_are_captured_at_default_threshold(tmp_path):
    store = TrainingCaptureStore(tmp_path / "auto_ads.jsonl")
    assessment = CandidateAssessment(0.85, ["ad_semantic_marker", "common_ad_dimensions"], "block")
    result = store.capture(ad_candidate(), assessment)
    assert result.saved


def test_medium_confidence_candidate_is_queued_without_becoming_a_label(tmp_path):
    review_path = tmp_path / "pending_review.jsonl"
    store = TrainingCaptureStore(tmp_path / "auto_ads.jsonl", review_path=review_path)
    assessment = CandidateAssessment(0.6, ["ad_semantic_marker"], "jev")
    result = store.capture(ad_candidate(), assessment)
    assert not result.saved
    assert result.queued
    assert result.reason == "queued_for_review"
    record = json.loads(review_path.read_text(encoding="utf-8"))
    assert record["question"]["label"] is None
    assert record["provenance"]["label_source"] == "unreviewed_candidate"


def test_manual_ad_label_is_saved_as_human_reviewed_data(tmp_path):
    reviewed_path = tmp_path / "reviewed.jsonl"
    store = TrainingCaptureStore(tmp_path / "auto_ads.jsonl", reviewed_path=reviewed_path)
    result = store.record_manual(ad_candidate(), label=0)
    assert result.saved
    assert result.reason == "manual_saved"
    record = json.loads(reviewed_path.read_text(encoding="utf-8"))
    assert record["question"]["label"] == 0
    assert record["provenance"]["label_source"] == "human_manual"
    assert record["state"]["resource_urls"] == ["https://ads.doubleclick.net/banner"]
    assert store.record_manual(ad_candidate(), label=0).reason == "manual_duplicate"


def test_rule_model_disagreement_is_never_auto_labeled(tmp_path):
    auto_path = tmp_path / "auto_ads.jsonl"
    review_path = tmp_path / "pending_review.jsonl"
    store = TrainingCaptureStore(auto_path, review_path=review_path)
    assessment = CandidateAssessment(0.85, ["known_ad_host"], "block")
    decision = Decision(
        "ad-1",
        Verdict.UNCERTAIN,
        {"block": 0.05, "allow": 0.9, "uncertain": 0.05},
        DecisionLevel.L2,
        diagnostics={"resolution": "review_disagreement", "model_verdict": "allow"},
    )
    result = store.capture(ad_candidate(), assessment, decision)
    assert not result.saved
    assert result.queued
    assert not auto_path.exists()
    record = json.loads(review_path.read_text(encoding="utf-8"))
    assert record["question"]["label"] is None
    assert record["provenance"]["selection_reason"] == "rule_model_disagreement"


def test_excluded_benchmark_domain_is_not_captured_or_queued(tmp_path):
    auto_path = tmp_path / "auto_ads.jsonl"
    review_path = tmp_path / "pending_review.jsonl"
    store = TrainingCaptureStore(
        auto_path,
        review_path=review_path,
        excluded_domains={"benchmark.test"},
    )
    item = ad_candidate()
    item.page = PageContext("https://sub.benchmark.test/", top_level_domain="sub.benchmark.test")
    assessment = CandidateAssessment(1.0, ["known_ad_host"], "block")
    result = store.capture(item, assessment)
    assert not result.saved
    assert not result.queued
    assert result.reason == "excluded_domain"
    assert not auto_path.exists()
    assert not review_path.exists()


def test_model_budget_deferred_candidate_does_not_fill_review_queue(tmp_path):
    review_path = tmp_path / "pending_review.jsonl"
    store = TrainingCaptureStore(tmp_path / "auto_ads.jsonl", review_path=review_path)
    decision = Decision(
        "candidate",
        Verdict.UNCERTAIN,
        {"block": 0.0, "allow": 0.0, "uncertain": 1.0},
        DecisionLevel.L0,
        diagnostics={"model_budget_deferred": True},
    )
    result = store.capture(
        ad_candidate(),
        CandidateAssessment(0.25, ["common_ad_dimensions"], "jev"),
        decision,
    )
    assert result.reason == "model_budget_deferred"
    assert not result.queued
    assert not review_path.exists()


def test_random_control_is_queued_without_becoming_a_label_and_has_own_quota(tmp_path):
    review_path = tmp_path / "pending_review.jsonl"
    store = TrainingCaptureStore(
        tmp_path / "auto_ads.jsonl",
        review_path=review_path,
        control_review_per_domain_limit=1,
    )
    item = ad_candidate()
    item.collection_stratum = "control"
    item.element_id = "ordinary-card"
    assessment = CandidateAssessment(0.0, [], "allow")
    first = store.capture(item, assessment)
    assert first.queued
    record = json.loads(review_path.read_text(encoding="utf-8"))
    assert record["question"]["label"] is None
    assert record["provenance"]["selection_reason"] == "random_control"
    second = ad_candidate()
    second.collection_stratum = "control"
    second.element_id = "ordinary-card-two"
    result = store.capture(second, assessment)
    assert result.reason == "control_domain_limit"
    assert not result.queued


def test_pending_candidate_review_becomes_human_training_record(tmp_path):
    review_path = tmp_path / "pending_review.jsonl"
    reviewed_path = tmp_path / "reviewed.jsonl"
    store = TrainingCaptureStore(
        tmp_path / "auto_ads.jsonl",
        review_path=review_path,
        reviewed_path=reviewed_path,
    )
    item = ad_candidate()
    item.collection_stratum = "control"
    item.element_id = "ordinary-card"
    queued = store.capture(item, CandidateAssessment(0.0, [], "allow"))
    pending = store.list_pending()
    assert [value["id"] for value in pending] == [queued.example_id]
    result = store.resolve_review(queued.example_id, 1)
    assert result["saved"]
    assert store.list_pending() == []
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    assert reviewed["id"] == queued.example_id
    assert reviewed["question"]["label"] == 1
    assert reviewed["provenance"]["label_source"] == "human_reviewed"
    assert reviewed["provenance"]["original_label_source"] == "unreviewed_candidate"
    assert store.resolve_review(queued.example_id, 0)["reason"] == "already_reviewed"
    assert store.stats()["review_examples"] == 0


def test_candidate_review_rejects_invalid_or_unknown_resolution(tmp_path):
    store = TrainingCaptureStore(tmp_path / "auto_ads.jsonl")
    assert store.resolve_review("missing", 7)["reason"] == "invalid_label"
    assert store.resolve_review("missing", 1)["reason"] == "not_found"
