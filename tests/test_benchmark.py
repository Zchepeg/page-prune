import json

from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateRanker
from evaluation.benchmark import (
    AblationEvaluator,
    AblationMode,
    BenchmarkCase,
    claim_eligibility,
    compare_mode_rows,
    domain_leakage,
    load_benchmark,
    public_report,
    training_domains,
    write_reports,
)
from jev.backend import BackendResult
from jev.decider import JevDecider
from jev.protocol import DecisionProtocol
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from probe.types import ProbeRecord


class TextBackend:
    def decide(self, state, question):
        payload = json.loads(state)
        if "probe" in payload:
            evidence = payload["probe"].get("evidence", [])
            probabilities = (
                {"ad": 0.9, "non_ad": 0.05, "insufficient_evidence": 0.05}
                if "probe_known_ad_host" in evidence
                else {"ad": 0.15, "non_ad": 0.15, "insufficient_evidence": 0.7}
            )
            return BackendResult(probabilities, DecisionLevel.L2, "text-test", {})
        text = payload["text"]
        if text == "paid":
            probabilities = {"ad": 0.9, "non_ad": 0.05, "insufficient_evidence": 0.05}
        elif text == "normal":
            probabilities = {"ad": 0.05, "non_ad": 0.9, "insufficient_evidence": 0.05}
        else:
            probabilities = {"ad": 0.15, "non_ad": 0.15, "insufficient_evidence": 0.7}
        return BackendResult(probabilities, DecisionLevel.L2, "text-test", {})


def candidate(identifier, domain, text, **values):
    return Candidate(
        identifier,
        PageContext(f"https://{domain}", top_level_domain=domain),
        text=text,
        **values,
    )


def evaluator():
    return AblationEvaluator(
        CandidateRanker(),
        JevDecider(TextBackend()),
        BlockingPolicy(),
        DecisionProtocol(),
    )


def probe_for(domain):
    before = {
        "width": 300,
        "height": 250,
        "visible": False,
        "in_viewport": False,
        "fixed": False,
        "child_count": 0,
        "resource_hosts": [],
        "connected": True,
    }
    after = {**before, "visible": True, "in_viewport": True, "child_count": 1, "resource_hosts": ["ads.doubleclick.net"]}
    return ProbeRecord.from_dict(
        {
            "probe_id": "benchmark-probe",
            "protocol_version": "ssa-probe-v1",
            "probe_type": "resource_association",
            "domain": domain,
            "candidate_fingerprint": "a" * 64,
            "seed_digest": "b" * 16,
            "detector_score": 0.72,
            "detector_route": "jev",
            "original_action": "allow",
            "sampling_rate": 0.005,
            "scheduled_delay_ms": 900,
            "started_at": "2026-10-10T00:00:00+00:00",
            "completed_at": "2026-10-10T00:00:01+00:00",
            "before": before,
            "after": after,
            "delta": {
                "width": 0,
                "height": 0,
                "visibility_changed": True,
                "viewport_changed": True,
                "children": 1,
                "new_resource_hosts": ["ads.doubleclick.net"],
                "mutation_count": 3,
                "disconnected": False,
            },
        }
    )


def test_benchmark_loader_rejects_domain_mismatch(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "domain": "expected.test",
                "label": "ad",
                "candidate": candidate("c1", "other.test", "paid").to_dict(),
            }
        ),
        encoding="utf-8",
    )
    try:
        load_benchmark(path)
    except ValueError as exc:
        assert "domain does not match" in str(exc)
    else:
        raise AssertionError("domain mismatch must fail")


def test_domain_leakage_uses_training_source_domains(tmp_path):
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"source": "seen.test"}) + "\n", encoding="utf-8")
    cases = [BenchmarkCase("one", "seen.test", 0, candidate("c1", "seen.test", "paid"))]
    result = domain_leakage(cases, training_domains([training]))
    assert not result["domain_disjoint"]
    assert result["overlap_domains"] == ["seen.test"]


def test_domain_leakage_rejects_parent_and_subdomain_overlap():
    cases = [BenchmarkCase("one", "news.example.test", 0, candidate("c1", "news.example.test", "paid"))]
    result = domain_leakage(cases, {"example.test"})
    assert not result["domain_disjoint"]


def test_claim_eligibility_rejects_smoke_dataset():
    cases = [BenchmarkCase("one", "new.test", 0, candidate("c1", "new.test", "paid"))]
    result = claim_eligibility(
        cases,
        domain_leakage(cases, set()),
        {
            "annotation_method": "synthetic_smoke",
            "labels_frozen_before_evaluation": True,
            "provenance": "test",
        },
    )
    assert not result["eligible"]
    assert "independent_human_annotation_required" in result["reasons"]
    assert "fewer_than_300_examples" in result["reasons"]


def test_claim_eligibility_accepts_sufficient_independent_dataset():
    cases = [
        BenchmarkCase(
            f"case-{label}-{index}",
            f"domain-{index % 30}.test",
            label,
            candidate(f"c-{label}-{index}", f"domain-{index % 30}.test", "normal"),
        )
        for label in range(3)
        for index in range(100)
    ]
    result = claim_eligibility(
        cases,
        domain_leakage(cases, set()),
        {
            "annotation_method": "independent_human",
            "labels_frozen_before_evaluation": True,
            "provenance": "independent annotation round",
            "used_for_supervision": False,
        },
    )
    assert result["eligible"]
    assert result["reasons"] == []


def test_claim_eligibility_accepts_binary_human_ground_truth():
    cases = [
        BenchmarkCase(
            f"case-{label}-{index}",
            f"domain-{index % 30}.test",
            label,
            candidate(f"c-{label}-{index}", f"domain-{index % 30}.test", "normal"),
        )
        for label in range(2)
        for index in range(150)
    ]
    result = claim_eligibility(
        cases,
        domain_leakage(cases, set()),
        {
            "annotation_method": "independent_human",
            "ground_truth_task": "binary_ad_detection",
            "labels_frozen_before_evaluation": True,
            "provenance": "independent annotation round",
            "used_for_supervision": False,
        },
    )
    assert result["eligible"]
    assert result["requirements"]["required_labels"] == [0, 1]


def test_claim_eligibility_rejects_benchmark_used_for_supervision():
    cases = [
        BenchmarkCase(
            f"case-{label}-{index}",
            f"domain-{index % 30}.test",
            label,
            candidate(f"c-{label}-{index}", f"domain-{index % 30}.test", "normal"),
        )
        for label in range(2)
        for index in range(150)
    ]
    result = claim_eligibility(
        cases,
        domain_leakage(cases, set()),
        {
            "annotation_method": "independent_human",
            "ground_truth_task": "binary_ad_detection",
            "labels_frozen_before_evaluation": True,
            "provenance": "independent annotation round",
            "used_for_supervision": True,
        },
    )
    assert not result["eligible"]
    assert "benchmark_used_for_supervision" in result["reasons"]


def test_ablation_runs_all_modes_and_surfaces_metrics():
    cases = [
        BenchmarkCase(
            "ad",
            "ads.test",
            0,
            candidate("ad", "ads.test", "paid", resource_urls=["https://ads.doubleclick.net/banner"]),
        ),
        BenchmarkCase("normal", "normal.test", 1, candidate("normal", "normal.test", "normal", tag="ARTICLE")),
        BenchmarkCase("unknown", "unknown.test", 2, candidate("unknown", "unknown.test", "unknown", width=300, height=250)),
    ]
    results = evaluator().evaluate(cases)
    assert set(results) == {mode.value for mode in AblationMode}
    assert results["model_only"]["three_class_accuracy"] == 1.0
    assert results["model_only"]["binary_ad_metrics"]["f1"] == 1.0
    assert results["jev_ad_protocol"]["latency_ms"]["p95"] >= 0.0


def test_report_writer_creates_json_csv_and_markdown(tmp_path):
    cases = [BenchmarkCase("normal", "normal.test", 1, candidate("normal", "normal.test", "normal"))]
    report = {
        "domain_leakage": domain_leakage(cases, set()),
        "modes": evaluator().evaluate(cases, [AblationMode.MODEL_ONLY]),
    }
    paths = write_reports(tmp_path, report)
    assert all(path.exists() for path in paths)
    assert "model_only" in paths[1].read_text(encoding="utf-8-sig")
    assert "# Ablation report" in paths[2].read_text(encoding="utf-8")


def test_mode_comparison_counts_corrections_and_regressions():
    baseline = {
        "rows": [
            {"id": "ad", "label": 0, "prediction": 2, "action": "allow"},
            {"id": "content", "label": 1, "prediction": 1, "action": "allow"},
        ]
    }
    candidate_mode = {
        "rows": [
            {"id": "ad", "label": 0, "prediction": 0, "action": "block"},
            {"id": "content", "label": 1, "prediction": 0, "action": "hide"},
        ]
    }
    result = compare_mode_rows(baseline, candidate_mode)
    assert result["changed_predictions"] == 2
    assert result["changed_actions"] == 2
    assert result["corrected_predictions"] == 1
    assert result["regressed_predictions"] == 1
    assert result["recovered_ads"] == 1
    assert result["added_destructive_false_positives"] == 1


def test_public_report_removes_rows_paths_and_domain_names():
    report = {
        "format_version": 1,
        "benchmark": "C:/private/benchmark.jsonl",
        "benchmark_metadata": {
            "annotation_method": "independent_human",
            "ground_truth_task": "binary_ad_detection",
            "labels_frozen_before_evaluation": True,
            "provenance": "C:/private/source",
        },
        "domain_leakage": {
            "evaluation_domains": 30,
            "training_domains": 40,
            "overlap_count": 1,
            "overlap_domains": ["private.test"],
            "domain_disjoint": False,
        },
        "modes": {"rules_only": {"examples": 1, "rows": [{"id": "secret"}]}},
    }
    result = public_report(report)
    serialized = json.dumps(result)
    assert result["privacy_level"] == "aggregate_only"
    assert "benchmark" not in result
    assert "rows" not in result["modes"]["rules_only"]
    assert "private.test" not in serialized
    assert "C:/private" not in serialized


def test_ssa_mode_reports_review_discovery_without_auto_blocking():
    domain = "missed-ad.test"
    case = BenchmarkCase(
        "missed-ad",
        domain,
        0,
        candidate("missed-ad", domain, "unknown", width=300, height=250),
        probe_for(domain),
    )
    result = evaluator().evaluate([case], [AblationMode.PROTOCOL_WITH_SSA])[AblationMode.PROTOCOL_WITH_SSA.value]
    assert result["ssa"]["flagged"] == 1
    assert result["ssa"]["ad_discovery_recall"] == 1.0
    assert result["rows"][0]["action"] == "allow"
    assert result["ssa"]["available_probes"] == 1
    assert result["ssa"]["assessed_probes"] == 1
    assert result["ssa"]["execution_status"] == "executed"


def test_ssa_without_observations_is_not_executed():
    case = BenchmarkCase("no-probe", "example.test", 1, candidate("no-probe", "example.test", "unknown"))
    result = evaluator().evaluate([case], [AblationMode.PROTOCOL_WITH_SSA])[AblationMode.PROTOCOL_WITH_SSA.value]
    assert result["ssa"]["execution_status"] == "not_executed"
    assert result["ssa"]["assessed_probes"] == 0
    assert result["routing"]["model_calls"] == sum(row["model_called"] for row in result["rows"])
