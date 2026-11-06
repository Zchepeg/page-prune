import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

import pytest

from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateRanker
from jev.backend import HeuristicBackend
from jev.decider import JevDecider
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
        service.process_many([candidate('one')])
    counts = service.performance_stats()['counters']
    assert counts['failed_requests'] == counts['model_errors'] == 1
    assert counts['requests_inflight'] == 0
    assert counts['candidates_decided'] == 0


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
