from __future__ import annotations

import hashlib
import math
import queue
import threading
import time
from collections import Counter, OrderedDict, deque

from collector.sanitizer import sanitize_candidate
from collector.state_builder import build_state_payload
from runtime.state_serialization import serialize_state


class LightweightShadow:
    def __init__(self, decider, policy, state_schema='candidate-features-v2', queue_size=64, dedup_size=512, ttl_seconds=900, max_age_seconds=30, on_decision=None):
        if queue_size < 1 or dedup_size < 1 or ttl_seconds <= 0 or max_age_seconds <= 0:
            raise ValueError('shadow budgets must be positive')
        self.decider = decider
        self.policy = policy
        self.schema = state_schema
        self.queue = queue.Queue(maxsize=queue_size)
        self.dedup_size = dedup_size
        self.ttl_seconds = ttl_seconds
        self.max_age_seconds = max_age_seconds
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.seen = OrderedDict()
        self.counts = Counter()
        self.actions = Counter()
        self.transitions = Counter()
        self.probability_bins = Counter()
        self.latencies = deque(maxlen=512)
        self.worker_cpu_ms = 0.0
        self.last_error_type = None
        self.on_decision = on_decision
        self.worker = threading.Thread(target=self._run, name='jev-lightweight-shadow', daemon=True)
        self.worker.start()

    def offer(self, candidate, assessment, primary_action):
        with self.lock:
            self.counts['observed'] += 1
            if self.stop.is_set():
                self.counts['closed_dropped'] += 1
                return False
            if assessment.route != 'jev':
                self.counts['clear_skipped'] += 1
                return False
        try:
            candidate = sanitize_candidate(candidate, max_text=500, max_html=0)
            evidence = tuple(assessment.evidence)
            state = serialize_state(build_state_payload(candidate, list(evidence), self.schema))
            key = hashlib.sha256(state.encode('utf-8')).hexdigest()
            now = time.monotonic()
            with self.lock:
                if self.stop.is_set():
                    self.counts['closed_dropped'] += 1
                    return False
                if key in self.seen and now-self.seen[key] < self.ttl_seconds:
                    self.counts['duplicate_skipped'] += 1
                    return False
                try:
                    self.queue.put_nowait((candidate, state, evidence, str(primary_action), key, now))
                except queue.Full:
                    self.counts['queue_dropped'] += 1
                    return False
                self.seen[key] = now
                self.seen.move_to_end(key)
                while len(self.seen) > self.dedup_size:
                    self.seen.popitem(last=False)
                self.counts['queued'] += 1
            return True
        except Exception as error:
            with self.lock:
                self.counts['input_errors'] += 1
                self.last_error_type = type(error).__name__
            return False

    def _run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                candidate, state, evidence, primary, key, queued_at = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                if time.monotonic()-queued_at > self.max_age_seconds:
                    with self.lock:
                        self.counts['stale_dropped'] += 1
                        self.seen.pop(key, None)
                    continue
                started, cpu_started = time.perf_counter(), time.thread_time()
                decision = self.decider.decide(candidate.candidate_id, state, list(evidence))
                action = self.policy.apply(candidate, decision).action.value
                elapsed = (time.perf_counter()-started)*1000
                cpu_ms = (time.thread_time()-cpu_started)*1000
                with self.lock:
                    self.counts['processed'] += 1
                    self.counts['action_disagreements'] += int(action != primary)
                    self.counts['potential_new_actions'] += int(primary not in ('hide','block') and action in ('hide','block'))
                    self.actions[action] += 1
                    self.transitions[f'{primary}->{action}'] += 1
                    self.probability_bins[min(9, max(0, int(decision.probabilities['block']*10)))] += 1
                    self.latencies.append(elapsed)
                    self.worker_cpu_ms += cpu_ms
                if self.on_decision is not None:
                    try:
                        self.on_decision(candidate, decision, action, primary)
                    except Exception as error:
                        with self.lock:
                            self.counts['callback_errors'] += 1
                            self.last_error_type = type(error).__name__
            except Exception as error:
                with self.lock:
                    self.counts['model_errors'] += 1
                    self.last_error_type = type(error).__name__
                    self.seen.pop(key, None)
            finally:
                self.queue.task_done()

    def wait_idle(self, timeout=5):
        with self.queue.all_tasks_done:
            return self.queue.all_tasks_done.wait_for(lambda:self.queue.unfinished_tasks == 0, timeout=timeout)

    def close(self, timeout=5):
        with self.lock:
            self.stop.set()
        self.worker.join(timeout=timeout)
        return not self.worker.is_alive()

    def stats(self):
        with self.lock:
            timings = sorted(self.latencies)
            counts = {key:self.counts[key] for key in ('observed','clear_skipped','queued','processed','duplicate_skipped','queue_dropped','stale_dropped','closed_dropped','input_errors','model_errors','callback_errors','action_disagreements','potential_new_actions')}
            return {'scope':'shadow-only; no commands, training labels or raw candidate persistence; disagreements are not ground truth', 'counters':counts, 'would_actions':dict(self.actions), 'primary_to_shadow_actions':dict(self.transitions), 'ad_probability_bins':dict(self.probability_bins), 'decision_ms':{'samples':len(timings),'mean':sum(timings)/len(timings) if timings else None,'p95':timings[math.ceil(.95*len(timings))-1] if timings else None}, 'worker_cpu_ms':round(self.worker_cpu_ms,3),'queue_capacity':self.queue.maxsize,'queue_depth':self.queue.qsize(),'dedup_entries':len(self.seen),'dedup_capacity':self.dedup_size,'timing_capacity':512,'last_error_type':self.last_error_type,'worker_alive':self.worker.is_alive()}
