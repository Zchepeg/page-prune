from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.annotation import load_jsonl
from scripts.build_evaluation_pool import EDGE, collect_page, overlaps, registrable_host, write_records
from service.config import load_config

SITES = '''atlasobscura.com mentalfloss.com sciencealert.com livescience.com space.com universetoday.com skyatnightmagazine.com astronomy.com discovermagazine.com popsci.com popularmechanics.com smithsonianmag.com nationalgeographic.com history.com thoughtco.com treehugger.com grist.org insideclimatenews.org carbonbrief.org undark.org aeon.co psyche.co nautil.us quantamagazine.org sciencenews.org scientificamerican.com newscientist.com inverse.com futurism.com bigthink.com salon.com slate.com vox.com reason.com prospect.org inthesetimes.com thenation.com jacobin.com theconversation.com theweek.com politico.com thehill.com axios.com semafor.com al-monitor.com middleeasteye.net france24.com rfi.fr swissinfo.ch swissinfo.ch/en irishtimes.com independent.ie scotsman.com heraldscotland.com standard.co.uk cityam.com euractiv.com balkaninsight.com bellingcat.com restofworld.org techspot.com notebookcheck.net anandtech.com tweaktown.com guru3d.com kitguru.net hexus.net overclock3d.net pcper.com videocardz.com wccftech.com neowin.net betanews.com ghacks.net androidpolice.com androidauthority.com 9to5google.com 9to5mac.com macrumors.com cultofmac.com appleinsider.com idownloadblog.com osnews.com itsfoss.com linuxiac.com linuxinsider.com omgubuntu.co.uk lwn.net phoronix.com gamingonlinux.com rockpapershotgun.com eurogamer.net pcgamesn.com polygon.com destructoid.com nintendolife.com pushsquare.com purexbox.com gamesradar.com gamedeveloper.com gamereactor.eu roadtovr.com uploadvr.com outsideonline.com gearjunkie.com adventure-journal.com singletracks.com bikeradar.com cyclingweekly.com runnersworld.com womensrunning.com food52.com seriouseats.com simplyrecipes.com thekitchn.com eater.com bonappetit.com epicurious.com smittenkitchen.com minimalistbaker.com wellplated.com downshiftology.com loveandlemons.com cookieandkate.com damndelicious.net spendwithpennies.com recipetineats.com''' .split()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(directory):
    if directory.exists():
        raise ValueError('refusing to overwrite blind study directory')
    sources = sorted(set((ROOT/'data/training').rglob('*.jsonl')) | set((ROOT/'data/evaluation').glob('*.jsonl')) | set((ROOT/'runs').rglob('pool*.jsonl')))
    known = set()
    for path in sources:
        for row in load_jsonl(path):
            candidates = [row.get('domain'), row.get('source'), row.get('candidate', {}).get('page', {}).get('top_level_domain')]
            for value in candidates:
                if isinstance(value, str) and re.fullmatch(r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', value):
                    known.add(value.lower().removeprefix('www.'))
    sites = sorted(set('https://'+site+'/' for site in SITES if '/' not in site))
    random.Random(20261101).shuffle(sites)
    sites = [site for site in sites if not overlaps(registrable_host(site), known)]
    model_paths = {'qwen-v2-seed42': 'runs/ownership-paired-v2-v1', 'qwen-v2-seed7': 'runs/ownership-v2-seed7', 'qwen-v3-seed42': 'runs/ownership-paired-v3-v1', 'qwen-v3-seed7': 'runs/ownership-v3-seed7', 'tfidf': 'runs/tfidf-baseline-v1'}
    models = {}
    for name, relative in model_paths.items():
        checkpoint = ROOT/relative
        files = [p for p in checkpoint.rglob('*') if p.is_file() and (p.suffix in ('.pt', '.safetensors', '.joblib', '.json') or 'tokenizer' in p.parts)]
        if not files:
            raise ValueError('missing frozen checkpoint: '+name)
        models[name] = {'checkpoint': relative, 'files': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}}
    config = load_config(ROOT/'config/local.yaml')
    code_files = sorted(p for folder in ('collector', 'detector', 'jev', 'policy', 'evaluation', 'runtime', 'modeling') for p in (ROOT/folder).rglob('*.py'))
    code_files.extend([Path(__file__).resolve(), ROOT/'scripts/build_evaluation_pool.py', ROOT/'scripts/review_evaluation_pool.py'])
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'preregistered_before_capture_and_labels', 'target_candidates': 400, 'per_domain': 10, 'minimum_binary_examples': 300, 'minimum_domains': 30, 'minimum_per_binary_label': 50, 'seed': 20261101, 'collection_budget_seconds': 420, 'timeout_ms': 8000, 'candidate_kinds': ['suspicious', 'control', 'neutral', 'ambiguous'], 'urls': sites, 'exclusion_domains': sorted(known), 'exclusion_sources': {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}, 'models': models, 'policy': config['policy'], 'detector': config['detector'], 'protocol': config.get('decision_protocol', {}), 'jev_thresholds': {k:config['jev'][k] for k in ('uncertainty_margin', 'min_confidence')}, 'code_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in code_files}, 'analysis': {'primary': 'paired page-action ad recall and non-ad action FPR for rules versus each frozen candidate cascade', 'secondary': 'classification precision/recall/F1, coverage, ECE, model-call fraction, backend and request timing reported separately', 'uncertainty': '10000 paired domain bootstrap draws, seed42; exact candidate McNemar supplementary, not cluster-adjusted', 'multiple_comparisons': 'five candidate comparisons are descriptive; no best-seed selection or significance claim without multiplicity control', 'inputs': 'each checkpoint native schema, calibrated saved temperature, with rule evidence; without-evidence diagnostic secondary', 'labels': 'independent human blind only; no machine suggestions or AI-generated ground truth; labels and benchmark frozen before inference', 'stop': 'candidate count or fixed time budget, never model results; insufficient binary support reported as pending, no weakened gates', 'deployment': 'no automatic promotion; no retraining or threshold search on this pool', 'new_domains': 'exclude all prior training, calibration, evaluation and candidate pools, including parent/subdomain matches; not a public-suffix or organizational independence guarantee'}}
    directory.mkdir(parents=True)
    (directory/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'prepared': True, 'eligible_sites': len(sites), 'excluded_domains': len(known), 'frozen_models': len(models), 'plan_sha256': digest(directory/'plan.json')}))


def collect(directory):
    from playwright.sync_api import sync_playwright

    plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    if (directory/'collection.json').exists() or (directory/'pool.jsonl').exists():
        raise ValueError('refusing to resume or overwrite frozen collection; use a separately preregistered supplement')
    for relative, expected in plan['code_sha256'].items():
        if digest(ROOT/relative) != expected:
            raise ValueError('code changed after registration: '+relative)
    screenshots = directory/'screenshots'
    screenshots.mkdir()
    records, pages, identifiers, domains = [], [], set(), set()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='jev-fresh-blind-') as profile:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(profile, executable_path=str(EDGE), headless=True, args=['--no-first-run', '--disable-background-networking'], viewport={'width':1440, 'height':1000}, locale='zh-CN')
            try:
                for index, url in enumerate(plan['urls']):
                    if time.monotonic()-started >= plan['collection_budget_seconds'] or len(records) >= plan['target_candidates']:
                        break
                    page = context.new_page()
                    try:
                        captured, status = collect_page(page, url, plan['timeout_ms'], plan['per_domain'], screenshots, plan['seed']+index, tuple(plan['candidate_kinds']))
                        if status['status'] is None or status['status'] >= 400 or overlaps(status['domain'], set(plan['exclusion_domains'])) or status['domain'] in domains:
                            status['excluded'] = True
                        else:
                            domains.add(status['domain'])
                            for row in captured:
                                if row['id'] not in identifiers:
                                    row['candidate']['html'] = ''
                                    records.append(row)
                                    identifiers.add(row['id'])
                    except Exception as error:
                        status = {'requested_url': url, 'error_type': type(error).__name__}
                    finally:
                        page.close()
                    pages.append(status)
                    write_records(directory/'pool.jsonl', records)
                    report = {'phase': 'unlabelled_capture_not_benchmark', 'plan_sha256': digest(directory/'plan.json'), 'captured': len(records), 'domains': len({r['domain'] for r in records}), 'elapsed_seconds': round(time.monotonic()-started, 2), 'pages': pages}
                    (directory/'collection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                    print(json.dumps({'pages':len(pages), 'captured':len(records), 'domains':report['domains'], 'elapsed_seconds':report['elapsed_seconds']}), flush=True)
            finally:
                context.close()
    write_records(directory/'pool.jsonl', records)
    manifest = {'plan_sha256': digest(directory/'plan.json'), 'pool_sha256': digest(directory/'pool.jsonl'), 'screenshots': {p.name:digest(p) for p in sorted(screenshots.glob('*.png'))}, 'captured':len(records), 'domains':len({r['domain'] for r in records}), 'annotations':0, 'inference_executed':False}
    (directory/'capture-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'capture_frozen':True, 'captured':len(records), 'domains':manifest['domains'], 'annotation_required':True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'collect'))
    parser.add_argument('--directory', type=Path, default=ROOT/'runs/fresh-blind-v1')
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else collect)(args.directory.resolve())


if __name__ == '__main__':
    main()
