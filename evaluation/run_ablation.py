from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from detector.candidate_ranker import CandidateRanker
from jev.backend import HeuristicBackend, QwenPointerBackend
from jev.decider import JevDecider
from jev.protocol import DecisionProtocol
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from service.config import load_config

from .benchmark import (
    AblationEvaluator,
    claim_eligibility,
    compare_mode_rows,
    domain_leakage,
    load_benchmark,
    public_report,
    training_domains,
    write_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JEV-Ad ablation evaluation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "config" / "config.yaml"))
    parser.add_argument("--out", default="artifacts/evaluation")
    parser.add_argument("--backend", choices=["configured", "heuristic"], default="configured")
    parser.add_argument("--metadata")
    parser.add_argument("--allow-domain-overlap", action="store_true")
    parser.add_argument("--public-summary", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    detector_config = config["detector"]
    jev_config = config["jev"]
    policy_config = config["policy"]
    protocol_config = config.get("decision_protocol", {})
    if args.backend == "heuristic":
        backend = HeuristicBackend()
    else:
        backend = QwenPointerBackend(
            project_root / jev_config["checkpoint"],
            jev_config.get("device", "cuda"),
            jev_config.get("quantization", ""),
            int(jev_config.get("gpu_memory_budget_mb", 0)),
            int(jev_config.get("inference_max_length", 0)),
        )
    cases = load_benchmark(args.data)
    metadata_path = Path(args.metadata) if args.metadata else Path(args.data).with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    source_paths = [project_root / path for path in config.get("auto_training", {}).get("training_sources", [])]
    leakage = domain_leakage(cases, training_domains(source_paths))
    if not leakage["domain_disjoint"] and not args.allow_domain_overlap:
        raise ValueError(f"evaluation domains overlap training data: {', '.join(leakage['overlap_domains'])}")
    evaluator = AblationEvaluator(
        CandidateRanker(detector_config["block_score"], detector_config["allow_score"]),
        JevDecider(
            backend,
            jev_config["uncertainty_margin"],
            float(jev_config.get("min_confidence", 0.0)),
        ),
        BlockingPolicy(
            policy_config["block_threshold"],
            policy_config["hide_threshold"],
            DecisionLevel.parse(policy_config["min_decision_level"]),
            set(policy_config["protected_tags"]),
            policy_config["fail_open"],
        ),
        DecisionProtocol(
            challenge_block_below=protocol_config.get("challenge_block_below", 0.98),
            challenge_allow_above=protocol_config.get("challenge_allow_above", 0.02),
            version=protocol_config.get("version", "jev-ad-decision-v1"),
            challenge_rule_blocks=bool(protocol_config.get("challenge_rule_blocks", True)),
        ),
    )
    modes = evaluator.evaluate(cases)
    report = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(Path(args.data).resolve()),
        "benchmark_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "backend": args.backend,
        "model_version": getattr(backend, "model_version", getattr(backend, "metadata", {}).get("architecture", "")),
        "protocol_version": protocol_config.get("version", "jev-ad-decision-v1"),
        "domain_leakage": leakage,
        "benchmark_metadata": metadata,
        "claim_eligibility": claim_eligibility(cases, leakage, metadata),
        "comparisons": {
            "cascade_vs_rules": compare_mode_rows(modes["rules_only"], modes["rule_model_cascade"]),
            "protocol_vs_rules": compare_mode_rows(modes["rules_only"], modes["jev_ad_protocol"]),
        },
        "modes": modes,
    }
    if args.public_summary:
        report = public_report(report)
    paths = write_reports(project_root / args.out, report)
    print(json.dumps({"reports": [str(path) for path in paths], "domain_leakage": leakage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
