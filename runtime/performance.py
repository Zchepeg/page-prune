from __future__ import annotations

import math
import threading
from collections import Counter, deque


class PerformanceMetrics:
    def __init__(self, window_size=512):
        if window_size < 1:
            raise ValueError('performance window must be positive')
        self._lock = threading.Lock()
        self._counts = Counter()
        self._sources = Counter()
        self._actions = Counter()
        self._request_ms = deque(maxlen=window_size)
        self._model_ms = deque(maxlen=window_size)
        self._prepare_ms = deque(maxlen=window_size)

    def begin(self, candidates):
        with self._lock:
            self._counts['requests_started'] += 1
            self._counts['requests_inflight'] += 1
            self._counts['candidates_received'] += candidates

    def request(self, elapsed_ms, failed=False):
        with self._lock:
            self._counts['requests'] += 1
            self._counts['requests_inflight'] -= 1
            self._counts['failed_requests'] += int(failed)
            self._request_ms.append(elapsed_ms)

    def preparation(self, elapsed_ms, warm_hits, demand, duplicates=0, deferred=0):
        with self._lock:
            self._prepare_ms.append(elapsed_ms)
            self._counts['warm_cache_hits'] += warm_hits
            self._counts['unique_model_demand'] += demand
            self._counts['batch_duplicate_model_candidates'] += duplicates
            self._counts['budget_deferred_candidates'] += deferred

    def model(self, states, elapsed_ms, failed=False):
        with self._lock:
            self._counts['model_batches'] += 1
            self._counts['model_states_submitted'] += states
            self._counts['model_errors'] += int(failed)
            self._model_ms.append(elapsed_ms)

    def result(self, source, action):
        with self._lock:
            self._counts['candidates_decided'] += 1
            self._sources[source] += 1
            self._actions[action] += 1

    def startup_deferred(self, count):
        with self._lock:
            self._counts['startup_deferred_candidates'] += count

    @staticmethod
    def _timing(values):
        if not values:
            return {'samples': 0, 'mean_ms': 0, 'p50_ms': 0, 'p95_ms': 0}
        ordered = sorted(values)
        return {'samples': len(values), 'mean_ms': round(sum(values)/len(values), 3), 'p50_ms': round(ordered[math.ceil(.5*len(values))-1], 3), 'p95_ms': round(ordered[math.ceil(.95*len(values))-1], 3)}

    def snapshot(self):
        names = ('requests_started', 'requests_inflight', 'requests', 'failed_requests', 'candidates_received', 'candidates_decided', 'warm_cache_hits', 'unique_model_demand', 'batch_duplicate_model_candidates', 'budget_deferred_candidates', 'startup_deferred_candidates', 'model_batches', 'model_states_submitted', 'model_errors')
        with self._lock:
            counts = {name: self._counts[name] for name in names}
            sources, actions = dict(self._sources), dict(self._actions)
            request, model, prepare = list(self._request_ms), list(self._model_ms), list(self._prepare_ms)
        received = counts['candidates_received']
        return {'scope': 'candidate decisions only; excludes probe inference, HTTP parsing/encryption and browser', 'counters': counts, 'sources': sources, 'actions': actions, 'warm_cache_hit_fraction': counts['warm_cache_hits']/received if received else 0, 'model_state_fraction': counts['model_states_submitted']/received if received else 0, 'request_window': self._timing(request), 'prepare_window': self._timing(prepare), 'model_batch_window': self._timing(model), 'window_capacity': self._request_ms.maxlen}
