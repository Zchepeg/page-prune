import json
from pathlib import Path

from feedback.event_store import EventStore


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "browser" / "extension"


def test_content_runtime_uses_incremental_bounded_scans():
    content = (EXTENSION / "content.js").read_text(encoding="utf-8")
    assert "MAX_CANDIDATES_PER_SCAN = 80" in content
    assert "MAX_DISCOVERED_PER_SCAN = 400" in content
    assert "selectCandidates(discovered)" in content
    assert "FULL_SCAN_INTERVAL_MS = 30000" in content
    assert "record.addedNodes" in content
    assert 'html: ""' in content
    assert "element.outerHTML" not in content
    assert ".slice(0, 320)" in content
    assert "direct_resource_urls: current.directUrls" in content
    assert "resource_ownership_known: true" in content
    assert "scanState.delete(entry.element)" in content
    assert "MAX_CONTROL_CANDIDATES_PER_PAGE = 4" in content
    assert 'consider(element, "control")' in content
    assert 'serialize(element, current, "manual")' in content


def test_background_runtime_uses_offline_backoff():
    background = (EXTENSION / "background.js").read_text(encoding="utf-8")
    assert "serviceRetryAt" in background
    assert "serviceFailureCount" in background
    assert "decision service retry backoff" in background
    assert "timeoutMs: 2500" in background


def test_event_store_batches_one_write(tmp_path):
    path = tmp_path / "events.jsonl"
    store = EventStore(path)
    store.append_many([{"id": 1}, {"id": 2}])
    assert [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()] == [1, 2]


def test_evaluation_collector_includes_randomized_control_stratum():
    collector = (ROOT / "scripts" / "build_evaluation_pool.py").read_text(encoding="utf-8")
    assert "const control = (hash >>> 0) % 20 === 0" in collector
    assert '("suspicious", "control", "neutral", "ambiguous")' in collector


def test_review_page_supports_candidate_and_probe_resolution():
    review = (ROOT / "service" / "review.html").read_text(encoding="utf-8")
    assert "/v1/training/review?limit=100" in review
    assert "/v1/training/review" in review
    assert "/v1/probes/review" in review
    assert "浏览器候选" in review
    assert "证据不足" in review
