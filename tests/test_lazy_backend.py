from concurrent.futures import ThreadPoolExecutor
import threading
import time

import jev.backend as backend_module
from jev.backend import BackendResult, LazyQwenPointerBackend
from jev.questions import AD_TREATMENT
from jev.types import DecisionLevel


def test_lazy_qwen_backend_loads_once_on_first_decision(monkeypatch):
    created = []

    class StubBackend:
        def __init__(self, *settings):
            created.append(settings)

        def decide_many(self, states, question):
            return [BackendResult({"ad": 0.2, "non_ad": 0.8}, DecisionLevel.L2, "stub", {}) for _ in states]

        def runtime_status(self):
            return {"backend": "stub", "loaded": True}

    monkeypatch.setattr(backend_module, "QwenPointerBackend", StubBackend)
    backend = LazyQwenPointerBackend("checkpoint")

    assert backend.runtime_status()["loaded"] is False
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: backend.decide_many(["state"], AD_TREATMENT), range(4)))

    assert len(created) == 1
    assert all(result[0].probabilities["non_ad"] == 0.8 for result in results)
    assert backend.runtime_status()["loaded"] is True


def wait_for_phase(backend, phase):
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        if backend.runtime_status()['phase'] == phase:
            return
        time.sleep(.005)
    raise AssertionError(f'backend did not reach {phase}')


def test_prepare_is_nonblocking_single_flight_and_warms_both_batch_sizes(monkeypatch):
    release, entered = threading.Event(), threading.Event()
    created, warmed = [], []
    class Slow:
        def __init__(self, *args):
            created.append(1)
            entered.set()
            assert release.wait(3)
        def decide_many(self, states, question):
            warmed.append(len(states))
            return []
        def runtime_status(self):
            return {'backend': 'stub', 'loaded': True}
    monkeypatch.setattr(backend_module, 'QwenPointerBackend', Slow)
    backend = LazyQwenPointerBackend('checkpoint')
    assert backend.prepare(['a', 'b', 'c'], AD_TREATMENT) is False
    assert entered.wait(1)
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert not any(executor.map(lambda _: backend.prepare(['a'], AD_TREATMENT), range(4)))
    release.set()
    wait_for_phase(backend, 'ready')
    assert backend.prepare(['a'], AD_TREATMENT) is True
    assert len(created) == 1 and warmed == [2, 1]


def test_prepare_failure_is_visible_and_backed_off(monkeypatch):
    attempts = []
    class Failure:
        def __init__(self, *args):
            attempts.append(1)
            raise RuntimeError('private checkpoint detail')
    monkeypatch.setattr(backend_module, 'QwenPointerBackend', Failure)
    backend = LazyQwenPointerBackend('checkpoint')
    assert backend.prepare(['a'], AD_TREATMENT) is False
    wait_for_phase(backend, 'failed')
    assert backend.prepare(['a'], AD_TREATMENT) is False
    status = backend.runtime_status()
    assert status['last_error_type'] == 'RuntimeError'
    assert 'private checkpoint detail' not in str(status)
    assert len(attempts) == 1
