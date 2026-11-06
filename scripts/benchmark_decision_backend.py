from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.state_serialization import serialize_state
from jev.questions import AD_TREATMENT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('tfidf', 'qwen'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/training/snapshots/calibration-supplement-v2/validation.jsonl')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    states = [serialize_state(json.loads(line)['state']) for line in Path(args.data).read_text(encoding='utf-8').splitlines()][:20]
    started = time.perf_counter()
    if args.backend == 'tfidf':
        from jev.lightweight_backend import TfidfBackend

        backend = TfidfBackend(args.checkpoint)
    else:
        from jev.backend import QwenPointerBackend

        backend = QwenPointerBackend(args.checkpoint, 'cuda', 'nf4', 4096, 384, 2)
    load_ms = (time.perf_counter()-started)*1000
    for state in states[:3]:
        backend.decide(state, AD_TREATMENT)
    timings = []
    for _ in range(3):
        for state in states:
            started = time.perf_counter()
            backend.decide(state, AD_TREATMENT)
            timings.append((time.perf_counter()-started)*1000)
    report = {'backend': args.backend, 'samples': len(states), 'repeats': 3, 'calls': len(timings), 'load_ms_including_backend_import': load_ms, 'mean_ms': statistics.mean(timings), 'median_ms': statistics.median(timings), 'p95_ms': sorted(timings)[math.ceil(.95*len(timings))-1], 'scope': 'serial batch1 backend only, 3 warmup calls; excludes service/browser/policy; same first20 heldout states', 'runtime': backend.runtime_status()}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
