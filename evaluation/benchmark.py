from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from collector.schemas import Candidate
from collector.state_builder import build_state
from detector.candidate_ranker import CandidateAssessment, CandidateRanker
from executor.blocker import ExecutionCommand
from jev.decider import JevDecider
from jev.protocol import DecisionPlan, DecisionProtocol, ProtocolRoute
from jev.types import Decision, DecisionLevel, Verdict
from policy.blocking_policy import BlockingPolicy
from probe.analyzer import ProbeAnalyzer
from probe.types import ProbeRecord
from jev.questions import PROBE_AD_TREATMENT

from .metrics import classification_metrics, expected_calibration_error


LABELS = {"ad": 0, "non_ad": 1, "insufficient_evidence": 2}


class AblationMode(StrEnum):
    RULES_ONLY = "rules_only"
    MODEL_ONLY = "model_only"
    CASCADE = "rule_model_cascade"
    PROTOCOL = "jev_ad_protocol"
    PROTOCOL_WITH_SSA = "jev_ad_protocol_with_ssa"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    domain: str
    label: int
    candidate: Candidate
    probe: ProbeRecord | None = None


def load_benchmark(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    identifiers: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                case_id = str(raw["id"]).strip()
                domain = str(raw["domain"]).lower().strip()
                raw_label = raw["label"]
                label = LABELS[raw_label] if isinstance(raw_label, str) else int(raw_label)
                candidate = Candidate.from_dict(raw["candidate"])
                probe = ProbeRecord.from_dict(raw["probe"]) if isinstance(raw.get("probe"), dict) else None
                if not case_id or not domain or label not in {0, 1, 2}:
                    raise ValueError("invalid id, domain, or label")
                if case_id in identifiers:
                    raise ValueError(f"duplicate id: {case_id}")
                if candidate.page.top_level_domain.lower().strip() != domain:
                    raise ValueError("candidate domain does not match case domain")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid benchmark record at {path}:{line_number}: {exc}") from exc
            identifiers.add(case_id)
            cases.append(BenchmarkCase(case_id, domain, label, candidate, probe))
    if not cases:
        raise ValueError(f"no benchmark cases found in {path}")
    return cases


def training_domains(paths: list[str | Path]) -> set[str]:
    domains: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                domain = str(value.get("source", "")).lower().strip()
                if domain:
                    domains.add(domain)
    return domains


def domain_leakage(cases: list[BenchmarkCase], known_training_domains: set[str]) -> dict[str, Any]:
    evaluation_domains = {case.domain for case in cases}
    overlap = sorted(
        domain
        for domain in evaluation_domains
        if any(
            domain == training_domain
            or domain.endswith("." + training_domain)
            or training_domain.endswith("." + domain)
            for training_domain in known_training_domains
        )
    )
    return {
        "evaluation_domains": len(evaluation_domains),
        "training_domains": len(known_training_domains),
        "overlap_count": len(overlap),
        "overlap_domains": overlap,
        "domain_disjoint": not overlap,
    }


def claim_eligibility(
    cases: list[BenchmarkCase],
    leakage: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    counts = Counter(case.label for case in cases)
    reasons: list[str] = []
    if not leakage.get("domain_disjoint", False):
        reasons.append("training_domain_overlap")
    if metadata.get("annotation_method") not in {
        "independent_human",
        "independent_expert_human",
    }:
        reasons.append("independent_human_annotation_required")
    if metadata.get("labels_frozen_before_evaluation") is not True:
        reasons.append("labels_not_declared_frozen")
    if not str(metadata.get("provenance", "")).strip():
        reasons.append("provenance_missing")
    if metadata.get("used_for_supervision") is True:
        reasons.append("benchmark_used_for_supervision")
    elif metadata.get("used_for_supervision") is not False:
        reasons.append("supervision_usage_not_declared")
    if len(cases) < 300:
        reasons.append("fewer_than_300_examples")
    if len({case.domain for case in cases}) < 30:
        reasons.append("fewer_than_30_domains")
    task = str(metadata.get("ground_truth_task", "three_class"))
    required_labels = (0, 1) if task == "binary_ad_detection" else (0, 1, 2)
    if any(counts.get(label, 0) < 50 for label in required_labels):
        reasons.append("fewer_than_50_examples_in_a_label")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "requirements": {
            "minimum_examples": 300,
            "minimum_domains": 30,
            "minimum_per_label": 50,
            "ground_truth_task": task,
            "required_labels": list(required_labels),
            "annotation_methods": [
                "independent_human",
                "independent_expert_human",
            ],
            "used_for_supervision": False,
        },
    }


class AblationEvaluator:
    def __init__(
        self,
        ranker: CandidateRanker,
        decider: JevDecider,
        policy: BlockingPolicy,
        protocol: DecisionProtocol,
        probe_analyzer: ProbeAnalyzer | None = None,
    ):
        self.ranker = ranker
        self.decider = decider
        self.policy = policy
        self.protocol = protocol
        self.probe_analyzer = probe_analyzer or ProbeAnalyzer()

    def evaluate(self, cases: list[BenchmarkCase], modes: list[AblationMode] | None = None) -> dict[str, Any]:
        selected = modes or list(AblationMode)
        if any(mode is not AblationMode.RULES_ONLY for mode in selected):
            candidate = cases[0].candidate
            self.decider.decide(candidate.candidate_id, build_state(candidate, []), [])
        return {mode.value: self._evaluate_mode(cases, mode) for mode in selected}

    def _evaluate_mode(self, cases: list[BenchmarkCase], mode: AblationMode) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for case in cases:
            started = time.perf_counter()
            assessment = self.ranker.assess(case.candidate)
            decision, source = self._decide(case.candidate, assessment, mode)
            policy_result = self.policy.apply(case.candidate, decision)
            command = ExecutionCommand.from_policy(case.candidate.candidate_id, policy_result)
            ssa_flagged = False
            ssa_route = ""
            if mode is AblationMode.PROTOCOL_WITH_SSA and case.probe and command.action.value in {"allow", "review"}:
                probe_assessment = self.probe_analyzer.assess(case.probe)
                ssa_route = probe_assessment.route
                ssa_flagged = probe_assessment.route != "discard"
                if probe_assessment.route == "model":
                    self.decider.decide(
                        case.probe.probe_id,
                        self.probe_analyzer.build_state(case.probe, probe_assessment),
                        probe_assessment.evidence,
                        PROBE_AD_TREATMENT,
                    )
            rows.append(
                {
                    "id": case.case_id,
                    "domain": case.domain,
                    "label": case.label,
                    "prediction": self._prediction(decision),
                    "block_probability": float(decision.probabilities.get("block", 0.0)),
                    "verdict": decision.verdict.value,
                    "action": command.action.value,
                    "source": source,
                    "protocol_route": str(decision.diagnostics.get("protocol_route", "")),
                    "model_called": mode is AblationMode.MODEL_ONLY or (mode is AblationMode.CASCADE and assessment.route == "jev") or (mode in {AblationMode.PROTOCOL, AblationMode.PROTOCOL_WITH_SSA} and self.protocol.plan(assessment).route is not ProtocolRoute.RULE),
                    "probe_available": case.probe is not None,
                    "ssa_assessed": mode is AblationMode.PROTOCOL_WITH_SSA and case.probe is not None and command.action.value in {"allow", "review"},
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "ssa_flagged": ssa_flagged,
                    "ssa_route": ssa_route,
                }
            )
        return self._summarize(rows)

    def _decide(
        self,
        candidate: Candidate,
        assessment: CandidateAssessment,
        mode: AblationMode,
    ) -> tuple[Decision, str]:
        if mode is AblationMode.RULES_ONLY:
            return self._rule_decision(candidate, assessment, allow_uncertain=True), "rules"
        if mode is AblationMode.MODEL_ONLY:
            return self.decider.decide(candidate.candidate_id, build_state(candidate, []), []), "model"
        if mode is AblationMode.CASCADE:
            if assessment.route == "jev":
                return self.decider.decide(
                    candidate.candidate_id,
                    build_state(candidate, assessment.evidence),
                    assessment.evidence,
                ), "model"
            return self._rule_decision(candidate, assessment), "rules"
        plan = self.protocol.plan(assessment)
        if plan.route is ProtocolRoute.RULE:
            return self._rule_decision(candidate, assessment, plan=plan), "rules"
        model = self.decider.decide(
            candidate.candidate_id,
            build_state(candidate, assessment.evidence),
            assessment.evidence,
        )
        resolution = self.protocol.resolve(plan, model)
        return resolution.decision, resolution.source

    def _rule_decision(
        self,
        candidate: Candidate,
        assessment: CandidateAssessment,
        allow_uncertain: bool = False,
        plan: DecisionPlan | None = None,
    ) -> Decision:
        if allow_uncertain and assessment.route == "jev":
            verdict = Verdict.UNCERTAIN
        elif assessment.route == "block":
            verdict = Verdict.BLOCK
        else:
            verdict = Verdict.ALLOW
        return Decision(
            candidate.candidate_id,
            verdict,
            {"block": assessment.score, "allow": 1.0 - assessment.score, "uncertain": 0.0},
            DecisionLevel.RULE,
            assessment.evidence,
            "rules-v1",
            "ad-classification-v1",
            {
                "protocol_version": self.protocol.version,
                "protocol_route": plan.route.value if plan else "ablation",
            },
        )

    @staticmethod
    def _prediction(decision: Decision) -> int:
        if decision.verdict is Verdict.BLOCK:
            return 0
        if decision.verdict is Verdict.ALLOW:
            return 1
        return 2

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        position = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[position]

    def _summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [int(row["label"]) for row in rows]
        predictions = [int(row["prediction"]) for row in rows]
        ad_labels = [label == 0 for label in labels]
        ad_predictions = [prediction == 0 for prediction in predictions]
        binary = classification_metrics(ad_labels, ad_predictions)
        binary["f1"] = (
            2 * binary["precision"] * binary["recall"] / (binary["precision"] + binary["recall"])
            if binary["precision"] + binary["recall"]
            else 0.0
        )
        confusion = [[0 for _ in range(3)] for _ in range(3)]
        for label, prediction in zip(labels, predictions):
            confusion[label][prediction] += 1
        destructive_false_positives = sum(
            row["label"] == 1 and row["action"] in {"block", "hide"} for row in rows
        )
        non_ads = sum(label == 1 for label in labels)
        latencies = [float(row["latency_ms"]) for row in rows]
        flagged = [row for row in rows if row["ssa_flagged"]]
        ad_count = sum(row["label"] == 0 for row in rows)
        flagged_ads = sum(row["label"] == 0 for row in flagged)
        return {
            "examples": len(rows),
            "domains": len({str(row["domain"]) for row in rows}),
            "labels": {str(key): value for key, value in Counter(labels).items()},
            "binary_ad_metrics": binary,
            "three_class_accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(rows),
            "coverage": sum(prediction != 2 for prediction in predictions) / len(rows),
            "ece_10_bin": expected_calibration_error(
                ad_labels,
                [float(row["block_probability"]) for row in rows],
            ),
            "page_action_false_positive_rate": destructive_false_positives / non_ads if non_ads else 0.0,
            "latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "p95": self._percentile(latencies, 0.95),
            },
            "confusion_actual_rows_predicted_columns": confusion,
            "sources": dict(Counter(str(row["source"]) for row in rows)),
            "routing": {
                "model_calls": sum(bool(row.get("model_called")) for row in rows),
                "model_call_rate": sum(bool(row.get("model_called")) for row in rows) / len(rows),
                "protocol_routes": dict(Counter(str(row.get("protocol_route", "")) for row in rows)),
                "ad_classifications_without_destructive_action": sum(row["prediction"] == 0 and row["action"] not in {"block", "hide"} for row in rows),
            },
            "ssa": {
                "available_probes": sum(bool(row.get("probe_available")) for row in rows),
                "assessed_probes": sum(bool(row.get("ssa_assessed")) for row in rows),
                "execution_status": "executed" if any(row.get("ssa_assessed") for row in rows) else "not_executed",
                "flagged": len(flagged),
                "review_rate": len(flagged) / len(rows),
                "ad_discovery_recall": flagged_ads / ad_count if ad_count else 0.0,
                "review_yield": flagged_ads / len(flagged) if flagged else 0.0,
                "routes": dict(Counter(str(row["ssa_route"]) for row in flagged)),
            },
            "rows": rows,
        }


def write_reports(output_dir: str | Path, report: dict[str, Any]) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "ablation.json"
    csv_path = output / "ablation.csv"
    markdown_path = output / "ablation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    modes = report["modes"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "mode",
            "examples",
            "domains",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "accuracy",
            "three_class_accuracy",
            "coverage",
            "ece_10_bin",
            "page_action_false_positive_rate",
            "mean_latency_ms",
            "p95_latency_ms",
            "ssa_flagged",
            "ssa_ad_discovery_recall",
            "ssa_review_yield",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, values in modes.items():
            binary = values["binary_ad_metrics"]
            writer.writerow(
                {
                    "mode": mode,
                    "examples": values["examples"],
                    "domains": values["domains"],
                    "precision": binary["precision"],
                    "recall": binary["recall"],
                    "f1": binary["f1"],
                    "false_positive_rate": binary["false_positive_rate"],
                    "accuracy": binary["accuracy"],
                    "three_class_accuracy": values["three_class_accuracy"],
                    "coverage": values["coverage"],
                    "ece_10_bin": values["ece_10_bin"],
                    "page_action_false_positive_rate": values["page_action_false_positive_rate"],
                    "mean_latency_ms": values["latency_ms"]["mean"],
                    "p95_latency_ms": values["latency_ms"]["p95"],
                    "ssa_flagged": values["ssa"]["flagged"],
                    "ssa_ad_discovery_recall": values["ssa"]["ad_discovery_recall"],
                    "ssa_review_yield": values["ssa"]["review_yield"],
                }
            )
    headers = [
        "Mode",
        "Precision",
        "Recall",
        "F1",
        "Rule/model coverage",
        "Action FPR",
        "ECE",
        "P95 ms",
        "SSA discoveries",
        "SSA yield",
    ]
    eligibility = report.get("claim_eligibility", {"eligible": False, "reasons": ["not_evaluated"]})
    lines = [
        "# Ablation report",
        "",
        f"Domain-disjoint: `{report['domain_leakage']['domain_disjoint']}`",
        f"Eligible for performance claims: `{eligibility['eligible']}`",
        f"Ineligibility reasons: `{', '.join(eligibility.get('reasons', [])) or 'none'}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for mode, values in modes.items():
        binary = values["binary_ad_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    f"{binary['precision']:.4f}",
                    f"{binary['recall']:.4f}",
                    f"{binary['f1']:.4f}",
                    f"{values['coverage']:.4f}",
                    f"{values['page_action_false_positive_rate']:.4f}",
                    f"{values['ece_10_bin']:.4f}",
                    f"{values['latency_ms']['p95']:.3f}",
                    str(values["ssa"]["flagged"]),
                    f"{values['ssa']['review_yield']:.4f}",
                ]
            )
            + " |"
        )
    comparisons = report.get("comparisons", {})
    if comparisons:
        lines.extend(
            [
                "",
                "## Contribution against rules-only",
                "",
                "| Comparison | Changed predictions | Corrected | Regressed | Recovered ads | Added destructive false positives |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, values in comparisons.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        str(values.get("changed_predictions", 0)),
                        str(values.get("corrected_predictions", 0)),
                        str(values.get("regressed_predictions", 0)),
                        str(values.get("recovered_ads", 0)),
                        str(values.get("added_destructive_false_positives", 0)),
                    ]
                )
                + " |"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path


def compare_mode_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, int]:
    baseline_rows = {str(row["id"]): row for row in baseline.get("rows", [])}
    candidate_rows = {str(row["id"]): row for row in candidate.get("rows", [])}
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("mode rows must contain the same sample ids")
    result = Counter()
    destructive = {"block", "hide"}
    for case_id, before in baseline_rows.items():
        after = candidate_rows[case_id]
        label = int(before["label"])
        before_correct = int(before["prediction"]) == label
        after_correct = int(after["prediction"]) == label
        if int(before["prediction"]) != int(after["prediction"]):
            result["changed_predictions"] += 1
        if str(before["action"]) != str(after["action"]):
            result["changed_actions"] += 1
        if not before_correct and after_correct:
            result["corrected_predictions"] += 1
        if before_correct and not after_correct:
            result["regressed_predictions"] += 1
        if label == 0 and int(before["prediction"]) != 0 and int(after["prediction"]) == 0:
            result["recovered_ads"] += 1
        if label == 0 and int(before["prediction"]) == 0 and int(after["prediction"]) != 0:
            result["lost_ads"] += 1
        before_destructive = str(before["action"]) in destructive
        after_destructive = str(after["action"]) in destructive
        if label == 1 and not before_destructive and after_destructive:
            result["added_destructive_false_positives"] += 1
        if label == 1 and before_destructive and not after_destructive:
            result["removed_destructive_false_positives"] += 1
    keys = (
        "changed_predictions",
        "changed_actions",
        "corrected_predictions",
        "regressed_predictions",
        "recovered_ads",
        "lost_ads",
        "added_destructive_false_positives",
        "removed_destructive_false_positives",
    )
    return {key: result[key] for key in keys}


def public_report(report: dict[str, Any]) -> dict[str, Any]:
    leakage = report.get("domain_leakage", {})
    metadata = report.get("benchmark_metadata", {})
    return {
        "format_version": report.get("format_version", 1),
        "privacy_level": "aggregate_only",
        "generated_at": report.get("generated_at", ""),
        "benchmark_sha256": report.get("benchmark_sha256", ""),
        "config_sha256": report.get("config_sha256", ""),
        "backend": report.get("backend", ""),
        "model_version": report.get("model_version", ""),
        "protocol_version": report.get("protocol_version", ""),
        "domain_leakage": {
            "evaluation_domains": leakage.get("evaluation_domains", 0),
            "training_domains": leakage.get("training_domains", 0),
            "overlap_count": leakage.get("overlap_count", 0),
            "domain_disjoint": leakage.get("domain_disjoint", False),
        },
        "benchmark_metadata": {
            "annotation_method": metadata.get("annotation_method", ""),
            "ground_truth_task": metadata.get("ground_truth_task", ""),
            "labels_frozen_before_evaluation": metadata.get("labels_frozen_before_evaluation", False),
            "used_for_supervision": metadata.get("used_for_supervision"),
        },
        "claim_eligibility": report.get("claim_eligibility", {}),
        "comparisons": report.get("comparisons", {}),
        "modes": {
            mode: {key: value for key, value in values.items() if key != "rows"}
            for mode, values in report.get("modes", {}).items()
        },
    }
