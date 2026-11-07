import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

import pytest

from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateRanker
from jev.backend import BackendResult, HeuristicBackend
from jev.decider import JevDecider
from jev.types import DecisionLevel
from local_security import LocalSessionManager
from policy.blocking_policy import BlockingPolicy
from runtime.decision_cache import DecisionCache
from runtime.performance import PerformanceMetrics
from service.api import make_handler
from service.orchestrator import AdBlockOrchestrator


def orchestrator(backend=None, budget=12):
    return AdBlockOrchestrator(CandidateRanker(), JevDecider(backend or HeuristicBackend()), BlockingPolicy(), max_model_candidates_per_batch=budget)


def candidate(identifier, slot='same'):
    return Candidate(identifier, PageContext('https://private.test', top_level_domain='private.test'), element_id=slot, width=300, height=250)


def test_metrics_distinguish_batch_reuse_from_warm_cache_hits():
    service = orchestrator()
    items = [candidate(str(i)) for i in range(3)]
    service.process_many(items)
    service.process_many(items)
    stats = service.performance_stats()
    counts = stats['counters']
    assert counts['requests'] == 2 and counts['candidates_received'] == 6
    assert counts['warm_cache_hits'] == 3
    assert counts['batch_duplicate_model_candidates'] == 2
    assert counts['model_batches'] == counts['model_states_submitted'] == 1
    assert counts['requests_inflight'] == 0
    assert stats['sources'] == {'jev': 1, 'cache': 5}
    assert stats['cache_lookups']['hits'] == 5
    assert stats['model_batch_window']['samples'] == 1
    assert 'private.test' not in json.dumps(stats)


def test_metrics_capture_budget_deferral_without_extra_model_work():
    service = orchestrator(budget=1)
    service.process_many([candidate(str(i), f'slot-{i}') for i in range(4)])
    counts = service.performance_stats()['counters']
    assert counts['unique_model_demand'] == 4
    assert counts['model_states_submitted'] == 1
    assert counts['budget_deferred_candidates'] == 3


def test_failed_model_records_failure_and_preserves_exception():
    class Failure:
        def decide(self, state, question):
            raise RuntimeError('test model failure')

    service = orchestrator(Failure())
    with pytest.raises(RuntimeError, match='test model failure'):
        service.process(candidate('one'))
    counts = service.performance_stats()['counters']
    assert counts['failed_requests'] == counts['model_errors'] == 1
    assert counts['requests_inflight'] == 0
    assert counts['candidates_decided'] == 0


@pytest.mark.parametrize('score', [float('nan'), float('inf'), 1.2, -0.1, 0.4])
def test_invalid_model_scores_do_not_emit_or_cache_decisions(score):
    class InvalidScores:
        def decide(self, state, question):
            return BackendResult({'ad': score, 'non_ad': 0.1}, DecisionLevel.L2, 'invalid', {})

    service = orchestrator(InvalidScores())
    with pytest.raises(ValueError, match='invalid probabilities'):
        service.process(candidate('one'))
    result = service.process_many([candidate('two')])[0]
    assert result['source'] == 'model_unavailable'
    assert result['command']['action'] == 'allow'
    assert result['training_capture']['reason'] == 'model_unavailable'

    stats = service.performance_stats()
    assert stats['counters']['model_errors'] == 2
    assert stats['counters']['candidates_decided'] == 1
    assert stats['cache_lookups']['resident_entries'] == 0


def test_short_batch_does_not_trigger_single_inference():
    class ShortBatch:
        def decide(self, state, question):
            raise AssertionError('unexpected single inference')

        def decide_many(self, states, question):
            assert len(states) == 2
            return [BackendResult({'ad': 0.9, 'non_ad': 0.1}, DecisionLevel.L2, 'short', {})]

    service = orchestrator(ShortBatch())
    block = Candidate('rule', PageContext('https://private.test'), resource_urls=['https://doubleclick.net/ads/banner'])
    results = service.process_many([candidate('one', 'a'), block, candidate('two', 'b')])
    assert [result['source'] for result in results] == ['model_unavailable', 'rules', 'model_unavailable']
    assert [result['command']['action'] for result in results] == ['allow', 'block', 'allow']
    assert service.performance_stats()['counters']['model_errors'] == 1


def test_invalid_model_result_does_not_discard_valid_batch_sibling():
    class MixedBatch:
        def decide_many(self, states, question):
            return [
                BackendResult({'ad': float('nan'), 'non_ad': 0.1}, DecisionLevel.L2, 'invalid', {}),
                BackendResult({'ad': 0.01, 'non_ad': 0.99}, DecisionLevel.L2, 'valid', {}),
            ]

    service = orchestrator(MixedBatch())
    results = service.process_many([candidate('bad', 'a'), candidate('good', 'b')])
    assert [result['source'] for result in results] == ['model_unavailable', 'jev']
    assert service.performance_stats()['cache_lookups']['resident_entries'] == 1


def test_batch_failure_cools_down_inference_without_delaying_rules_or_cache(monkeypatch):
    class Backend:
        calls = 0
        failed = False

        def decide_many(self, states, question):
            self.calls += 1
            if self.failed:
                raise RuntimeError('test backend unavailable')
            return [BackendResult({'ad': 0.01, 'non_ad': 0.99}, DecisionLevel.L2, 'test', {}) for _ in states]

    clock = [100.0]
    monkeypatch.setattr('service.orchestrator.monotonic', lambda: clock[0])
    backend = Backend()
    service = orchestrator(backend, budget=1)
    warm = candidate('warm', 'warm-slot')
    failed = candidate('failed', 'failed-slot')
    fresh = candidate('fresh', 'fresh-slot')
    block = Candidate('rule', PageContext('https://private.test'), resource_urls=['https://doubleclick.net/ads/banner'])
    assert service.process_many([warm])[0]['source'] == 'jev'
    backend.failed = True
    assert [item['source'] for item in service.process_many([failed, fresh])] == ['model_unavailable', 'model_unavailable']
    assert service.performance_stats()['model_batch_cooldown_ms'] == 5000

    backend.failed = False
    clock[0] = 104.0
    held = service.process_many([warm, block, fresh])
    assert [item['source'] for item in held] == ['cache', 'rules', 'model_unavailable']
    assert held[1]['command']['action'] == 'block'
    assert backend.calls == 2

    clock[0] = 105.0
    assert service.process_many([fresh])[0]['source'] == 'jev'
    assert backend.calls == 3
    assert service.performance_stats()['model_batch_cooldown_ms'] == 0


def test_only_one_batch_checks_recovery_after_cooldown(monkeypatch):
    recovery_started = threading.Event()
    release_recovery = threading.Event()

    class Backend:
        calls = 0

        def decide_many(self, states, question):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('test backend unavailable')
            if self.calls == 2:
                recovery_started.set()
                assert release_recovery.wait(timeout=5)
            return [BackendResult({'ad': 0.01, 'non_ad': 0.99}, DecisionLevel.L2, 'test', {}) for _ in states]

    clock = [100.0]
    monkeypatch.setattr('service.orchestrator.monotonic', lambda: clock[0])
    backend = Backend()
    service = orchestrator(backend)
    first = candidate('first', 'first-slot')
    fresh = candidate('fresh', 'fresh-slot')
    rule = Candidate('rule', PageContext('https://private.test'), resource_urls=['https://doubleclick.net/ads/banner'])
    assert service.process_many([first])[0]['source'] == 'model_unavailable'
    clock[0] = 105.0

    with ThreadPoolExecutor(max_workers=2) as pool:
        recovery = pool.submit(service.process_many, [first])
        try:
            assert recovery_started.wait(timeout=5)
            concurrent = pool.submit(service.process_many, [fresh, rule]).result(timeout=3)
            assert [row['source'] for row in concurrent] == ['model_unavailable', 'rules']
            assert concurrent[1]['command']['action'] == 'block'
            assert backend.calls == 2
        finally:
            release_recovery.set()
        assert recovery.result(timeout=5)[0]['source'] == 'jev'

    assert service.process_many([fresh])[0]['source'] == 'jev'
    assert backend.calls == 3


def test_metric_windows_are_bounded_and_concurrent_counts_are_exact():
    metrics = PerformanceMetrics(window_size=3)
    def record(_):
        metrics.begin(1)
        metrics.request(2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(100)))
    stats = metrics.snapshot()
    assert stats['counters']['requests'] == 100
    assert stats['counters']['requests_inflight'] == 0
    assert stats['request_window'] == {'samples': 3, 'mean_ms': 2, 'p50_ms': 2, 'p95_ms': 2}


def test_cache_stats_cover_expiration_and_eviction(monkeypatch):
    clock = [0]
    monkeypatch.setattr('runtime.decision_cache.time.monotonic', lambda: clock[0])
    cache = DecisionCache(max_size=1, ttl_seconds=5)
    assert cache.get_by_key('missing') is None
    cache.put_by_key('a', object())
    assert cache.get_by_key('a') is not None
    cache.put_by_key('b', object())
    clock[0] = 6
    assert cache.get_by_key('b') is None
    stats = cache.stats()
    assert stats['hits'] == 1 and stats['misses'] == 2
    assert stats['expired_misses'] == stats['evictions'] == 1
    assert stats['resident_entries'] == 0


def test_status_exposes_aggregate_performance(tmp_path):
    service = orchestrator()
    sessions = LocalSessionManager(registry_path=tmp_path/'paired.json')
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(service, sessions))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        connection.request('GET', '/v1/status')
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 200
        assert payload['performance']['counters']['requests'] == 0
        assert payload['performance']['cache_lookups']['resident_entries'] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
