from __future__ import annotations

from pathlib import Path

from detector.candidate_ranker import CandidateRanker
from feedback.active_learning import ActiveLearningSampler
from feedback.disagreement_store import DisagreementStore
from feedback.event_store import EventStore
from feedback.training_capture import TrainingCaptureStore
from jev.backend import HeuristicBackend, LazyQwenPointerBackend, QwenPointerBackend
from jev.decider import JevDecider
from jev.protocol import DecisionProtocol
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from probe.analyzer import ProbeAnalyzer
from probe.review_store import ProbeReviewStore
from probe.store import ProbeStore
from runtime.decision_cache import DecisionCache
from training.auto_trainer import AutoTrainer
from .config import load_config
from .orchestrator import AdBlockOrchestrator


def create_orchestrator(config_path: str | Path) -> AdBlockOrchestrator:
    project_root = Path(config_path).resolve().parent.parent
    config = load_config(config_path)
    detector = config["detector"]
    jev = config["jev"]
    policy = config["policy"]
    privacy = config["privacy"]
    runtime = config["runtime"]
    capture = config.get("training_capture", {})
    protocol = config.get("decision_protocol", {})
    active_learning = config.get("active_learning", {})
    shadow_audit = config.get("shadow_audit", {})
    auto_trainer = AutoTrainer(project_root, config.get("auto_training", {}), jev)
    if jev["backend"] == "qwen_pointer":
        checkpoint = Path(jev["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = project_root / checkpoint
        backend_type = LazyQwenPointerBackend if bool(jev.get("lazy_load", True)) else QwenPointerBackend
        backend = backend_type(
            checkpoint,
            jev.get("device", "cuda"),
            jev.get("quantization", ""),
            int(jev.get("gpu_memory_budget_mb", 0)),
            int(jev.get("inference_max_length", 0)),
            int(jev.get("micro_batch_size", 2)),
        )
    elif jev["backend"] == "heuristic":
        backend = HeuristicBackend()
    else:
        raise ValueError(f"unknown decision backend: {jev['backend']}")
    orchestrator = AdBlockOrchestrator(
        ranker=CandidateRanker(detector["block_score"], detector["allow_score"]),
        decider=JevDecider(
            backend,
            jev["uncertainty_margin"],
            float(jev.get("min_confidence", 0.0)),
        ),
        policy=BlockingPolicy(
            block_threshold=policy["block_threshold"],
            hide_threshold=policy["hide_threshold"],
            min_decision_level=DecisionLevel.parse(policy["min_decision_level"]),
            protected_tags=set(policy["protected_tags"]),
            fail_open=policy["fail_open"],
        ),
        cache=DecisionCache(runtime["cache_size"], runtime["cache_ttl_seconds"]),
        events=EventStore(project_root / "data/events/decisions.jsonl"),
        disagreements=DisagreementStore(
            project_root / protocol.get("disagreement_path", "data/events/disagreements.jsonl")
        ),
        protocol=DecisionProtocol(
            challenge_block_below=float(protocol.get("challenge_block_below", 0.98)),
            challenge_allow_above=float(protocol.get("challenge_allow_above", 0.02)),
            version=str(protocol.get("version", "jev-ad-decision-v1")),
            challenge_rule_blocks=bool(protocol.get("challenge_rule_blocks", True)),
        ),
        probe_store=ProbeStore(
            project_root / shadow_audit.get("path", "data/events/probes.jsonl"),
            enabled=bool(shadow_audit.get("enabled", False)),
        ),
        probe_analyzer=ProbeAnalyzer(
            review_threshold=float(shadow_audit.get("review_threshold", 0.15)),
            model_threshold=float(shadow_audit.get("model_threshold", 0.30)),
        ),
        probe_reviews=ProbeReviewStore(
            project_root / shadow_audit.get("review_path", "data/training/pending_probe_review.jsonl"),
            project_root / shadow_audit.get("reviewed_path", "data/training/reviewed_probes.jsonl"),
        ),
        training_capture=TrainingCaptureStore(
            path=project_root / capture.get("path", "data/training/auto_ads.jsonl"),
            review_path=project_root / capture.get("review_path", "data/training/pending_review.jsonl"),
            reviewed_path=project_root / capture.get("reviewed_path", "data/training/reviewed.jsonl"),
            enabled=bool(capture.get("enabled", False)),
            min_score=float(capture.get("min_score", 0.85)),
            review_min_score=float(capture.get("review_min_score", 0.35)),
            per_domain_limit=int(capture.get("per_domain_limit", 500)),
            review_per_domain_limit=int(capture.get("review_per_domain_limit", 200)),
            control_review_per_domain_limit=int(capture.get("control_review_per_domain_limit", 12)),
            max_text_length=int(capture.get("max_text_length", 240)),
            excluded_domains=capture.get("excluded_domains", []),
            sampler=ActiveLearningSampler(
                min_priority=float(active_learning.get("min_priority", 0.45)),
                block_score=float(detector["block_score"]),
                allow_score=float(detector["allow_score"]),
            ),
        ),
        auto_trainer=auto_trainer,
        max_text=privacy["max_text_length"],
        max_html=privacy["max_html_length"],
        max_model_candidates_per_batch=int(runtime.get("max_model_candidates_per_batch", 12)),
        defer_model_startup=bool(runtime.get('defer_model_startup', False)),
    )
    auto_trainer.maybe_start()
    return orchestrator
