from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev.backend import QwenPointerBackend
from jev.questions import AD_TREATMENT, TypedQuestion
from training.dataset import load_jsonl
from runtime.state_serialization import serialize_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--per-label', type=int, default=30)
    parser.add_argument('--out', default='runs/model-diagnosis.json')
    args = parser.parse_args()
    backend = QwenPointerBackend(args.checkpoint, 'cuda', 'nf4', 4096, 384, 2)
    report = {'metadata': backend.metadata, 'splits': {}}
    for split in ('train', 'validation'):
        examples = load_jsonl(Path(args.snapshot) / f'{split}.jsonl', 'binary_selective')
        selected, counts = [], Counter()
        for example in examples:
            if counts[example.label] < args.per_label:
                selected.append(example)
                counts[example.label] += 1
        result = {'available_labels': dict(Counter(e.label for e in examples)), 'sampled_labels': dict(counts), 'instructions': dict(Counter(e.instruction for e in examples)), 'variants': {}}
        for variant in ('training_input', 'runtime_instruction', 'without_evidence'):
            rows = []
            for example in selected:
                state = json.loads(example.state)
                if variant == 'without_evidence':
                    state['detector_evidence'] = []
                instruction = example.instruction if variant == 'training_input' else AD_TREATMENT.text
                question = TypedQuestion('diagnostic', instruction, backend.model_labels, 'diagnostic-v1')
                output = backend.decide(serialize_state(state), question)
                p = output.probabilities['ad']
                rows.append({'label': example.label, 'p_ad': p, 'truncated': output.diagnostics['truncated']})
            stats = {}
            for label in (0, 1):
                values = sorted(row['p_ad'] for row in rows if row['label'] == label)
                stats[str(label)] = {'n': len(values), 'min': min(values), 'median': values[len(values)//2], 'max': max(values), 'ad_argmax': sum(p > .5 for p in values), 'ad_selective': sum(p >= .55 for p in values)} if values else {'n': 0}
            result['variants'][variant] = {'by_label': stats, 'truncated': sum(row['truncated'] for row in rows)}
        report['splits'][split] = result
        print(json.dumps({split: result}), flush=True)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
