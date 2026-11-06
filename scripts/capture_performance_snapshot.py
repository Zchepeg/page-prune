from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('refusing to overwrite a performance snapshot')
    with urlopen('http://127.0.0.1:8765/v1/status', timeout=5) as response:
        raw = response.read(131073)
    if len(raw) > 131072:
        raise ValueError('status response exceeds snapshot size limit')
    status = json.loads(raw)
    runtime = status['decision_runtime']
    report = {'scope': 'aggregate service performance only', 'service': status['service'], 'performance': status['performance'], 'decision_runtime': {key: runtime[key] for key in ('backend', 'loaded', 'quantization', 'inference_max_length', 'micro_batch_size', 'cuda_allocated_mb', 'cuda_reserved_mb', 'phase', 'ready', 'load_ms', 'warmup_ms', 'last_error_type') if key in runtime}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'snapshot': str(args.out), 'requests': report['performance']['counters']['requests'], 'model_loaded': runtime['loaded']}))


if __name__ == '__main__':
    main()
