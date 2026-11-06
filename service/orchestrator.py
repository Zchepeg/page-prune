from __future__ import annotations

import time
from dataclasses import replace

from collector.sanitizer import sanitize_candidate
from collector.schemas import Candidate
from collector.state_builder import build_state
from detector.candidate_ranker import CandidateRanker
from executor.blocker import ExecutionCommand
from feedback.disagreement_store import DisagreementStore
from feedback.event_store import EventStore
from feedback.training_capture import TrainingCaptureStore
from jev.decider import JevDecider
from jev.protocol import DecisionPlan, DecisionProtocol, ProtocolRoute
from jev.questions import PROBE_AD_TREATMENT
from jev.types import Decision, DecisionLevel, Verdict
from policy.blocking_policy import BlockingPolicy
from probe.analyzer import ProbeAnalyzer
from probe.review_store import ProbeReviewStore
from probe.store import ProbeStore
from probe.types import ProbeRecord
from runtime.decision_cache import DecisionCache
from runtime.performance import PerformanceMetrics
from training.auto_trainer import AutoTrainer


class AdBlockOrchestrator:
    def __init__(
        self,
        ranker: CandidateRanker,
        decider: JevDecider,
        policy: BlockingPolicy,
        cache: DecisionCache | None = None,
        events: EventStore | None = None,
        disagreements: DisagreementStore | None = None,
        training_capture: TrainingCaptureStore | None = None,
        auto_trainer: AutoTrainer | None = None,
        protocol: DecisionProtocol | None = None,
        probe_store: ProbeStore | None = None,
        probe_analyzer: ProbeAnalyzer | None = None,
        probe_reviews: ProbeReviewStore | None = None,
        max_text: int = 500,
        max_html: int = 2000,
        max_model_candidates_per_batch: int = 12,
        defer_model_startup: bool = False,
    ):
        self.ranker = ranker
        self.decider = decider
        self.policy = policy
        self.cache = cache or DecisionCache()
        self.events = events or EventStore()
        self.disagreements = disagreements or DisagreementStore()
        self.training_capture = training_capture
        self.auto_trainer = auto_trainer
        self.protocol = protocol or DecisionProtocol()
        self.probe_store = probe_store or ProbeStore()
        self.probe_analyzer = probe_analyzer or ProbeAnalyzer()
        self.probe_reviews = probe_reviews
        self.max_text = max_text
        self.max_html = max_html
        self.max_model_candidates_per_batch = max(1, max_model_candidates_per_batch)
        self.performance = PerformanceMetrics()
        self.defer_model_startup = defer_model_startup

    def process(self, candidate: Candidate, event_sink: list[dict] | None = None) -> dict:
        started = time.perf_counter()
        self.performance.begin(1)
        failed = True
        try:
            result = self._process_single(candidate, event_sink)
            failed = False
            return result
        finally:
            self.performance.request((time.perf_counter()-started)*1000, failed)

    def _process_single(self, candidate, event_sink=None):
        started = time.perf_counter()
        candidate = sanitize_candidate(candidate, self.max_text, self.max_html)
        assessment = self.ranker.assess(candidate)
        key = self.cache.key(candidate)
        cached = self.cache.get_by_key(key)
        plan = None if cached is not None else self.protocol.plan(assessment)
        prepare_ms = (time.perf_counter()-started)*1000
        self.performance.preparation(prepare_ms, int(cached is not None), int(plan is not None and plan.route is not ProtocolRoute.RULE))
        starting = bool(self.defer_model_startup and plan is not None and plan.route is not ProtocolRoute.RULE and not self.decider.prepare_many([(candidate.candidate_id, build_state(candidate, assessment.evidence), assessment.evidence)]))
        if starting:
            self.performance.startup_deferred(1)
        return self._process_prepared(candidate, assessment, event_sink=event_sink, cached=cached, plan=plan, cache_key=key, cache_checked=True, model_allowed=not starting, startup_deferred=starting)

    def _process_prepared(
        self,
        candidate: Candidate,
        assessment,
        event_sink: list[dict] | None = None,
        cached: Decision | None = None,
        plan: DecisionPlan | None = None,
        model_allowed: bool = True,
        model_decision: Decision | None = None,
        latency_offset_ms: float = 0.0,
        cache_key: str | None = None,
        cache_checked: bool = False,
        startup_deferred: bool = False,
    ) -> dict:
        started = time.perf_counter()
        if not cache_checked:
            cache_key = cache_key or self.cache.key(candidate)
            cached = self.cache.get_by_key(cache_key)
        if cached is not None:
            decision = replace(cached, candidate_id=candidate.candidate_id)
            source = "cache"
        else:
            plan = plan or self.protocol.plan(assessment)
            if plan.route is ProtocolRoute.RULE:
                decision = self._rule_decision(candidate, assessment, plan)
                source = "rules"
            elif not model_allowed:
                decision = self._budget_decision(candidate, assessment, plan, startup_deferred)
                source = 'model_starting' if startup_deferred else "model_budget"
            else:
                if model_decision is None:
                    state = build_state(candidate, assessment.evidence)
                    model_started = time.perf_counter()
                    model_failed = True
                    try:
                        model_decision = self.decider.decide(candidate.candidate_id, state, assessment.evidence)
                        model_failed = False
                    finally:
                        self.performance.model(1, (time.perf_counter()-model_started)*1000, model_failed)
                resolution = self.protocol.resolve(plan, model_decision)
                decision = resolution.decision
                source = resolution.source
                self.disagreements.append(candidate, assessment, resolution)
        if source not in {"model_budget", 'model_starting'}:
            self.cache.put_by_key(cache_key or self.cache.key(candidate), decision)
        capture = self.training_capture.capture(candidate, assessment, decision) if self.training_capture and source != 'model_starting' else None
        if capture and capture.saved and self.auto_trainer:
            self.auto_trainer.maybe_start()
        policy_result = self.policy.apply(candidate, decision)
        command = ExecutionCommand.from_policy(candidate.candidate_id, policy_result)
        latency_ms = round(latency_offset_ms + (time.perf_counter() - started) * 1000, 3)
        result = {
            "command": command.to_dict(),
            "decision": decision.to_dict(),
            "source": source,
            "latency_ms": latency_ms,
            "training_capture": capture.to_dict() if capture else {"saved": False, "reason": 'model_starting' if source == 'model_starting' else "not_configured", "example_id": ""},
            "detector": {
                "score": round(assessment.score, 6),
                "route": assessment.route,
            },
        }
        event = {
                "candidate_id": candidate.candidate_id,
                "domain": candidate.page.top_level_domain,
                "action": command.action.value,
                "verdict": decision.verdict.value,
                "level": decision.level.name,
                "source": source,
                "latency_ms": latency_ms,
                "detector_route": assessment.route,
                "detector_score": round(assessment.score, 6),
                "evidence": assessment.evidence,
                "capture_saved": bool(capture and capture.saved),
                "capture_reason": capture.reason if capture else "not_configured",
            }
        if event_sink is None:
            self.events.append(event)
        else:
            event_sink.append(event)
        self.performance.result(source, command.action.value)
        return result

    def _budget_decision(self, candidate: Candidate, assessment, plan: DecisionPlan, starting=False) -> Decision:
        return Decision(
            candidate_id=candidate.candidate_id,
            verdict=Verdict.UNCERTAIN,
            probabilities={"block": 0.0, "allow": 0.0, "uncertain": 1.0},
            level=DecisionLevel.L0,
            evidence=assessment.evidence,
            model_version='model-starting-v1' if starting else "model-budget-v1",
            question_version="ad-classification-v1",
            diagnostics={
                "protocol_version": self.protocol.version,
                "protocol_route": plan.route.value,
                "protocol_reason": plan.reason,
                "resolution": 'model_loading_deferred' if starting else "model_budget_deferred",
                "model_budget_deferred": not starting,
                "model_loading_deferred": starting,
                "retry_after_ms": 2000,
            },
        )

    def _rule_decision(
        self,
        candidate: Candidate,
        assessment,
        plan: DecisionPlan,
    ) -> Decision:
        verdict = plan.rule_verdict or Verdict.UNCERTAIN
        return Decision(
            candidate_id=candidate.candidate_id,
            verdict=verdict,
            probabilities={"block": assessment.score, "allow": 1 - assessment.score, "uncertain": 0.0},
            level=DecisionLevel.RULE,
            evidence=assessment.evidence,
            model_version="rules-v1",
            question_version="ad-classification-v1",
            diagnostics={
                "protocol_version": self.protocol.version,
                "protocol_route": plan.route.value,
                "protocol_reason": plan.reason,
                "rule_verdict": verdict.value,
                "resolution": "rule_fast_path",
            },
        )

    def process_many(self, candidates: list[Candidate]) -> list[dict]:
        started = time.perf_counter()
        self.performance.begin(len(candidates))
        failed = True
        try:
            results = self._process_batch(candidates)
            failed = False
            return results
        finally:
            self.performance.request((time.perf_counter()-started)*1000, failed)

    def _process_batch(self, candidates):
        prepare_started = time.perf_counter()
        events: list[dict] = []
        prepared = []
        model_groups = {}
        for index, raw_candidate in enumerate(candidates):
            candidate = sanitize_candidate(raw_candidate, self.max_text, self.max_html)
            assessment = self.ranker.assess(candidate)
            cache_key = self.cache.key(candidate)
            cached = self.cache.get_by_key(cache_key)
            plan = None if cached is not None else self.protocol.plan(assessment)
            prepared.append((candidate, assessment, cached, plan, cache_key))
            if cached is None and plan is not None and plan.route is not ProtocolRoute.RULE:
                model_groups.setdefault(cache_key, []).append(index)
        ranked_groups = sorted(
            model_groups.items(),
            key=lambda item: (-max(prepared[index][1].score for index in item[1]), min(item[1])),
        )
        allowed_groups = dict(ranked_groups[: self.max_model_candidates_per_batch])
        allowed = {index for indices in allowed_groups.values() for index in indices}
        representatives = [indices[0] for indices in allowed_groups.values()]
        duplicate_indices = {index for indices in allowed_groups.values() for index in indices[1:]}
        requests = [
            (
                prepared[index][0].candidate_id,
                build_state(prepared[index][0], prepared[index][1].evidence),
                prepared[index][1].evidence,
            )
            for index in representatives
        ]
        self.performance.preparation((time.perf_counter()-prepare_started)*1000, sum(item[2] is not None for item in prepared), len(model_groups), sum(len(indices)-1 for indices in model_groups.values()), sum(len(indices) for key, indices in ranked_groups if key not in allowed_groups))
        starting_indices = set()
        if requests and self.defer_model_startup and not self.decider.prepare_many(requests):
            starting_indices = allowed
            self.performance.startup_deferred(len(starting_indices))
            allowed = set()
            requests = []
            duplicate_indices = set()
        model_started = time.perf_counter()
        model_failed = True
        try:
            batch_decisions = self.decider.decide_many(requests) if requests else []
            model_failed = False
        finally:
            if requests:
                self.performance.model(len(requests), (time.perf_counter()-model_started)*1000, model_failed)
        model_latency_ms = (time.perf_counter() - model_started) * 1000
        model_decisions = {}
        for (key, indices), decision in zip(allowed_groups.items(), batch_decisions):
            for index in indices:
                model_decisions[index] = replace(decision, candidate_id=prepared[index][0].candidate_id)
        results = [
            self._process_prepared(
                candidate,
                assessment,
                events,
                cached,
                plan,
                index in allowed,
                model_decisions.get(index),
                model_latency_ms if index in allowed else 0.0,
                cache_key,
                index not in duplicate_indices,
                index in starting_indices,
            )
            for index, (candidate, assessment, cached, plan, cache_key) in enumerate(prepared)
        ]
        self.events.append_many(events)
        return results

    def training_stats(self) -> dict:
        capture = self.training_capture.stats() if self.training_capture else {"enabled": False, "examples": 0, "domains": 0}
        auto_training = self.auto_trainer.status() if self.auto_trainer else {"enabled": False, "status": "not_configured"}
        return {"capture": capture, "auto_training": auto_training}

    def runtime_stats(self) -> dict:
        return self.decider.runtime_status()

    def performance_stats(self) -> dict:
        return {**self.performance.snapshot(), 'cache_lookups': self.cache.stats(), 'defer_model_startup': self.defer_model_startup}

    def record_manual(self, candidate: Candidate, label: int = 0) -> dict:
        candidate = sanitize_candidate(candidate, self.max_text, self.max_html)
        if not self.training_capture:
            return {"saved": False, "queued": False, "reason": "not_configured", "example_id": ""}
        result = self.training_capture.record_manual(candidate, label)
        if result.saved and self.auto_trainer:
            self.auto_trainer.maybe_start()
        self.events.append(
            {
                "candidate_id": candidate.candidate_id,
                "domain": candidate.page.top_level_domain,
                "action": "manual_label",
                "label": label,
                "source": "human_manual",
                "capture_saved": result.saved,
                "capture_reason": result.reason,
            }
        )
        return result.to_dict()

    def record_probe(self, value: dict) -> dict:
        if not self.probe_store.enabled:
            return self.probe_store.append(value)
        record = ProbeRecord.from_dict(value)
        assessment = self.probe_analyzer.assess(record)
        decision = None
        if assessment.route == "model":
            state = self.probe_analyzer.build_state(record, assessment)
            decision = self.decider.decide(record.probe_id, state, assessment.evidence, PROBE_AD_TREATMENT)
        stored = self.probe_store.append(value)
        review = {"queued": False, "reason": "discarded", "review_id": ""}
        if assessment.route != "discard" and self.probe_reviews:
            review = self.probe_reviews.queue(record, assessment, decision)
        return {
            **stored,
            "analysis": assessment.to_dict(),
            "decision": decision.to_dict() if decision else None,
            "review": review,
        }

    def probe_stats(self) -> dict:
        reviews = self.probe_reviews.stats() if self.probe_reviews else {"pending": 0, "reviewed": 0}
        return {**self.probe_store.stats(), "reviews": reviews}

    def list_probe_reviews(self, limit: int = 50) -> list[dict]:
        return self.probe_reviews.list_pending(limit) if self.probe_reviews else []

    def resolve_probe_review(self, review_id: str, label: int) -> dict:
        if not self.probe_reviews:
            return {"saved": False, "reason": "not_configured", "review_id": review_id}
        return self.probe_reviews.resolve(review_id, label)

    def list_training_reviews(self, limit: int = 50) -> list[dict]:
        return self.training_capture.list_pending(limit) if self.training_capture else []

    def resolve_training_review(self, review_id: str, label: int) -> dict:
        if not self.training_capture:
            return {"saved": False, "reason": "not_configured", "review_id": review_id}
        result = self.training_capture.resolve_review(review_id, label)
        if result.get("saved") and self.auto_trainer:
            self.auto_trainer.maybe_start()
        self.events.append(
            {
                "candidate_id": review_id,
                "action": "review_label",
                "label": label,
                "source": "human_reviewed",
                "capture_saved": bool(result.get("saved")),
                "capture_reason": str(result.get("reason", "")),
            }
        )
        return result
