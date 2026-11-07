from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateRanker
from jev.backend import BackendResult, HeuristicBackend
from jev.decider import JevDecider
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from runtime.decision_cache import DecisionCache
from service.orchestrator import AdBlockOrchestrator


def make_orchestrator():
    return AdBlockOrchestrator(
        CandidateRanker(),
        JevDecider(HeuristicBackend()),
        BlockingPolicy(),
    )


def test_clear_ad_is_blocked_without_jev():
    item = Candidate(
        "ad-1",
        PageContext("https://news.test", top_level_domain="news.test"),
        tag="IFRAME",
        element_id="top-ad",
        resource_urls=["https://ads.doubleclick.net/banner"],
        visible=True,
    )
    result = make_orchestrator().process(item)
    assert result["source"] == "rules"
    assert result["command"]["action"] == "block"


def test_ambiguous_candidate_uses_jev_and_fails_open_or_hides():
    item = Candidate(
        "maybe-1",
        PageContext("https://news.test", top_level_domain="news.test"),
        width=300,
        height=250,
        visible=True,
    )
    result = make_orchestrator().process(item)
    assert result["source"] == "jev"
    assert result["command"]["action"] in {"allow", "hide"}


def test_cached_decision_uses_current_candidate_id():
    orchestrator = make_orchestrator()
    page = PageContext("https://news.test", top_level_domain="news.test")
    first = Candidate("first", page, element_id="ad", resource_urls=["https://ads.doubleclick.net/x"])
    second = Candidate("second", page, element_id="ad", resource_urls=["https://ads.doubleclick.net/x"])
    orchestrator.process(first)
    result = orchestrator.process(second)
    assert result["source"] == "cache"
    assert result["command"]["candidate_id"] == "second"


class CountingBackend:
    def __init__(self):
        self.calls = 0

    def decide(self, state, question):
        self.calls += 1
        return BackendResult(
            {"ad": 0.05, "non_ad": 0.9, "insufficient_evidence": 0.05},
            DecisionLevel.L2,
            "counting",
            {},
        )


class BatchCountingBackend:
    def __init__(self):
        self.batch_calls = 0
        self.state_counts = []

    def decide(self, state, question):
        raise AssertionError("single inference must not be used")

    def decide_many(self, states, question):
        self.batch_calls += 1
        self.state_counts.append(len(states))
        return [
            BackendResult(
                {"ad": 0.05, "non_ad": 0.9, "insufficient_evidence": 0.05},
                DecisionLevel.L2,
                "batch-counting",
                {},
            )
            for _ in states
        ]


class KeyCountingCache(DecisionCache):
    def __init__(self):
        super().__init__()
        self.key_calls = 0

    def key(self, candidate):
        self.key_calls += 1
        return super().key(candidate)


def test_process_many_limits_model_work_and_does_not_cache_deferred_decisions():
    backend = CountingBackend()
    orchestrator = AdBlockOrchestrator(
        CandidateRanker(),
        JevDecider(backend),
        BlockingPolicy(),
        max_model_candidates_per_batch=2,
    )
    page = PageContext("https://news.test", top_level_domain="news.test")
    candidates = [
        Candidate(f"candidate-{index}", page, element_id=f"slot-{index}", width=300, height=250)
        for index in range(5)
    ]
    results = orchestrator.process_many(candidates)
    assert backend.calls == 2
    assert [result["source"] for result in results].count("model_budget") == 3
    assert all(result["command"]["action"] == "allow" for result in results[2:])
    result = orchestrator.process(candidates[-1])
    assert result["source"] == "jev"
    assert backend.calls == 3


def test_process_many_batches_and_deduplicates_identical_states():
    backend = BatchCountingBackend()
    orchestrator = AdBlockOrchestrator(
        CandidateRanker(),
        JevDecider(backend),
        BlockingPolicy(),
        max_model_candidates_per_batch=3,
    )
    page = PageContext("https://news.test", top_level_domain="news.test")
    candidates = [
        Candidate(f"candidate-{index}", page, element_id="same-slot", width=300, height=250)
        for index in range(3)
    ]
    results = orchestrator.process_many(candidates)
    assert backend.batch_calls == 1
    assert backend.state_counts == [1]
    assert [result["source"] for result in results] == ["jev", "cache", "cache"]


def test_decisions_distinguish_text_tail_and_model_geometry():
    backend = BatchCountingBackend()
    orchestrator = AdBlockOrchestrator(CandidateRanker(), JevDecider(backend), BlockingPolicy())
    page = PageContext("https://news.test", top_level_domain="news.test")
    prefix = "x" * 170
    candidates = [
        Candidate("first", page, text=prefix + " first", width=300.1, height=250),
        Candidate("second", page, text=prefix + " second", width=300.1, height=250),
        Candidate("third", page, text=prefix + " first", width=300.4, height=250),
    ]
    assert len({orchestrator.cache.key(candidate) for candidate in candidates}) == 1

    results = orchestrator.process_many(candidates)

    assert backend.state_counts == [3]
    assert [result["source"] for result in results] == ["jev", "jev", "jev"]
    assert all(orchestrator.process(candidate)["source"] == "cache" for candidate in candidates)


def test_process_many_hashes_each_candidate_once():
    backend = BatchCountingBackend()
    cache = KeyCountingCache()
    orchestrator = AdBlockOrchestrator(
        CandidateRanker(),
        JevDecider(backend),
        BlockingPolicy(),
        cache=cache,
        max_model_candidates_per_batch=3,
    )
    page = PageContext("https://news.test", top_level_domain="news.test")
    candidates = [
        Candidate(f"candidate-{index}", page, element_id="same-slot", width=300, height=250)
        for index in range(3)
    ]

    orchestrator.process_many(candidates)

    assert cache.key_calls == len(candidates)


def test_async_startup_is_not_cached_or_captured_and_retries_when_ready():
    class Starting(CountingBackend):
        ready = False
        def prepare(self, states, question):
            return self.ready
    class Capture:
        calls = 0
        def capture(self, *args):
            self.calls += 1
            return None
    backend, capture = Starting(), Capture()
    service = AdBlockOrchestrator(CandidateRanker(), JevDecider(backend), BlockingPolicy(), training_capture=capture, defer_model_startup=True)
    page = PageContext('', top_level_domain='example.test')
    candidates = [Candidate(str(i), page, element_id='same', width=300, height=250) for i in range(3)]
    waiting = service.process_many(candidates)
    assert all(row['source'] == 'model_starting' and row['command']['action'] == 'allow' for row in waiting)
    assert service.cache.stats()['resident_entries'] == 0
    assert backend.calls == capture.calls == 0
    assert service.performance_stats()['counters']['startup_deferred_candidates'] == 3
    backend.ready = True
    ready = service.process_many(candidates)
    assert ready[0]['source'] == 'jev'
    assert backend.calls == 1 and capture.calls == 3


def test_single_candidate_startup_deferral_is_not_cached():
    class Starting(CountingBackend):
        def prepare(self, states, question):
            return False
    backend = Starting()
    service = AdBlockOrchestrator(CandidateRanker(), JevDecider(backend), BlockingPolicy(), defer_model_startup=True)
    result = service.process(Candidate('one', PageContext('', top_level_domain='example.test'), width=300, height=250))
    assert result['source'] == 'model_starting'
    assert service.cache.stats()['resident_entries'] == backend.calls == 0


def test_startup_failure_preserves_rules_and_recovers_without_caching_failure():
    class StartupBackend(CountingBackend):
        phase = "raises"

        def prepare(self, states, question):
            if self.phase == "raises":
                raise RuntimeError("test preparation error")
            return self.phase == "ready"

        def runtime_status(self):
            return {"phase": self.phase}

    backend = StartupBackend()
    service = AdBlockOrchestrator(
        CandidateRanker(), JevDecider(backend), BlockingPolicy(),
        defer_model_startup=True, max_model_candidates_per_batch=1,
    )
    page = PageContext("https://news.test", top_level_domain="news.test")
    one = Candidate("one", page, element_id="slot-a", width=300, height=250)
    two = Candidate("two", page, element_id="slot-b", width=300, height=250)
    rule = Candidate("rule", page, resource_urls=["https://doubleclick.net/ads/banner"])

    failed = service.process_many([one, rule, two])
    assert [row["source"] for row in failed] == ["model_unavailable", "rules", "model_unavailable"]
    assert [row["command"]["action"] for row in failed] == ["allow", "block", "allow"]
    assert service.process(one)["source"] == "model_unavailable"
    backend.phase = "failed"
    assert service.process(two)["source"] == "model_unavailable"
    assert service.cache.stats()["resident_entries"] == 1
    counts = service.performance_stats()["counters"]
    assert counts["model_states_submitted"] == counts["startup_deferred_candidates"] == backend.calls == 0

    backend.phase = "warming"
    assert service.process(one)["source"] == "model_starting"
    backend.phase = "ready"
    assert service.process(one)["source"] == "jev"
    assert backend.calls == 1
