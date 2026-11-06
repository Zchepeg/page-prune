from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.sanitizer import sanitize_candidate
from collector.state_builder import build_state_payload
from jev.lightweight_backend import TfidfBackend
from local_security import LocalSessionManager
from runtime.lightweight_shadow import LightweightShadow
from scripts.browser_scenarios import page_counts
from scripts.evaluate_fresh_blind import make_evaluator
from scripts.paired_browser_e2e import EDGE, EXTENSION, E2EOrchestrator, prepare_extension, start_server, wait_for_extension_id
from service.api import make_handler

SITES = ('https://www.python.org/','https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API','https://www.bbc.com/','https://www.theguardian.com/international','https://www.bilibili.com/','https://github.com/login')


def memory_mb():
    if sys.platform != 'win32':
        return {'rss_mb':None,'peak_rss_mb':None,'scope':'Windows process memory counters unavailable'}
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong),*[(name,ctypes.c_size_t) for name in ('peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    psapi = ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'rss_mb':round(counters.rss/1024**2,3),'peak_rss_mb':round(counters.peak_rss/1024**2,3),'scope':'Python service/driver process only, excludes browser child processes and GPU'}


class ShadowService(E2EOrchestrator):
    def __init__(self, evaluator, observer):
        super().__init__()
        self.evaluator = evaluator
        self.observer = observer
        self.review = deque(maxlen=100)
        self.lock = threading.Lock()
        self.request_ms = deque(maxlen=512)
        self.primary_commands = 0

    def process_many(self, candidates):
        started = time.perf_counter()
        results = []
        for raw in candidates:
            candidate = sanitize_candidate(raw,max_text=500,max_html=0)
            assessment = self.evaluator.ranker.assess(candidate)
            decision = self.evaluator._rule_decision(candidate,assessment,allow_uncertain=True)
            policy = self.evaluator.policy.apply(candidate,decision)
            result = {
                'command': {
                    'candidate_id': candidate.candidate_id,
                    'action': policy.action.value,
                    'reason': policy.reason,
                },
                'decision': decision.to_dict(),
                'source': 'rules',
                'latency_ms': 0,
                'training_capture': {
                    'saved': False,
                    'queued': False,
                    'reason': 'isolated_shadow_no_training',
                },
                'detector': {'score': assessment.score, 'route': assessment.route},
            }
            results.append(result)
            if self.observer:
                self.observer.offer(candidate,assessment,policy.action.value)
        with self.lock:
            self.request_ms.append((time.perf_counter()-started)*1000)
            self.primary_commands += len(results)
        return results

    def capture(self, candidate, decision, action, primary):
        if action in ('hide','block') and primary not in ('hide','block'):
            with self.lock:
                self.review.append({
                    'candidate_id': candidate.candidate_id,
                    'state': build_state_payload(candidate, decision.evidence),
                    'would_action': action,
                    'primary_action': primary,
                    'p_ad': decision.probabilities['block'],
                    'human_label': None,
                })

    def performance_stats(self):
        with self.lock:
            values = sorted(self.request_ms)
            primary = {'requests':len(values),'candidate_commands':self.primary_commands,'mean_ms':sum(values)/len(values) if values else None,'p95_ms':values[max(0,math.ceil(.95*len(values))-1)] if values else None}
        return {'primary':primary,'lightweight_shadow':self.observer.stats() if self.observer else {'enabled':False}}


def run(output, shadow_enabled, sites=SITES):
    if output.exists():
        raise ValueError('refusing to overwrite shadow run')
    output.mkdir(parents=True)
    (output/'screenshots').mkdir()
    plan = json.loads((ROOT/'runs/fresh-blind-review-v1/plan.json').read_text(encoding='utf-8'))
    before = memory_mb()
    load_started = time.perf_counter()
    backend = TfidfBackend(ROOT/plan['models']['tfidf']['checkpoint']) if shadow_enabled else None
    from jev.backend import HeuristicBackend
    evaluator = make_evaluator(plan,backend or HeuristicBackend(),'candidate-features-v2')
    service_adapter = ShadowService(evaluator,None)
    observer = LightweightShadow(evaluator.decider,evaluator.policy,on_decision=service_adapter.capture) if shadow_enabled else None
    service_adapter.observer = observer
    load_ms = (time.perf_counter()-load_started)*1000
    loaded_memory = memory_mb()
    results, captured = [], []
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='jev-shadow-browser-') as temporary:
        temporary = Path(temporary)
        sessions = LocalSessionManager(registry_path=temporary/'pairs.json')
        server, worker = start_server(make_handler(service_adapter,sessions))
        extension = prepare_extension(EXTENSION,temporary,server.server_port)
        script = extension/'content.js'
        text = script.read_text(encoding='utf-8')
        text = text.replace(
            'nodesById.set(candidateId, element);',
            'nodesById.set(candidateId, element); '
            'const shadowKeys = (element.getAttribute("data-jev-shadow-key") || "").split(" "); '
            'element.setAttribute("data-jev-shadow-key", [...shadowKeys.slice(-15), candidateId].join(" "));',
        )
        script.write_text(text,encoding='utf-8')
        try:
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    str(temporary / 'profile'),
                    executable_path=str(EDGE),
                    headless=True,
                    args=[
                        '--no-first-run',
                        '--disable-background-networking',
                        f'--disable-extensions-except={extension}',
                        f'--load-extension={extension}',
                    ],
                    viewport={'width': 1440, 'height': 1000},
                    locale='zh-CN',
                )
                try:
                    identifier = wait_for_extension_id(context)
                    options = context.new_page()
                    options.goto(f'chrome-extension://{identifier}/options.html')
                    options.locator('#code').fill(sessions.pairing_code)
                    options.locator('button').click()
                    options.locator('#status').filter(has_text='已绑定').wait_for(timeout=10000)
                    for index,url in enumerate(sites):
                        page = context.new_page()
                        page.set_default_timeout(5000)
                        row = {'requested_url':url}
                        try:
                            response = page.goto(url,wait_until='domcontentloaded',timeout=12000)
                            row['http_status'] = response.status if response else None
                            if response is None or response.status >= 400:
                                row['excluded_reason'] = 'http_error_or_missing_response'
                            else:
                                page.wait_for_timeout(3500)
                                row['before'] = page_counts(page)
                                page.evaluate('window.scrollTo(0,Math.min(1200,document.body.scrollHeight*.4))')
                                page.wait_for_timeout(2500)
                                row['after'] = page_counts(page)
                                if observer:
                                    row['shadow_drained'] = observer.wait_idle(timeout=5)
                                with service_adapter.lock:
                                    outcomes = list(service_adapter.review)
                                    service_adapter.review.clear()
                                for item in outcomes[:5]:
                                    identifier = item['candidate_id']
                                    selector = f'[data-jev-shadow-key~="{identifier}"]'
                                    locator = next(
                                        (match for frame in page.frames if (match := frame.locator(selector)).count() == 1),
                                        page.locator(selector),
                                    )
                                    image = output/'screenshots'/f'review-{len(captured)}.png'
                                    try:
                                        locator.screenshot(path=str(image),timeout=1500)
                                        item['screenshot'] = str(image)
                                        item['still_visible'] = locator.is_visible()
                                    except Exception:
                                        item['screenshot'] = ''
                                        item['still_visible'] = None
                                    if item['still_visible']:
                                        context_image = image.with_name(f'{image.stem}-context.png')
                                        try:
                                            page.screenshot(path=str(context_image), style=f'{selector} {{ outline: 4px solid #f50 !important; outline-offset: -4px !important; }}', timeout=2000)
                                            item['context_screenshot'] = str(context_image)
                                        except Exception:
                                            item['context_screenshot'] = ''
                                    captured.append(item)
                                row['shadow_new_actions_sent_to_browser'] = 0
                        except Exception as error:
                            row['error_type'] = type(error).__name__
                        finally:
                            page.close()
                        results.append(row)
                        (output/'progress.json').write_text(json.dumps({'pages':len(results),'shadow_enabled':shadow_enabled,'stats':service_adapter.performance_stats()},indent=2),encoding='utf-8')
                        print(json.dumps({'page':index+1,'http_status':row.get('http_status'),'error_type':row.get('error_type'),'shadow_processed':observer.stats()['counters']['processed'] if observer else 0}),flush=True)
                finally:
                    context.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
            if observer:
                observer.close()
    wall_seconds, cpu_seconds = time.perf_counter()-wall_started,time.process_time()-cpu_started
    report = {
        'scope': (
            'isolated anonymous public-page shadow run; ordinary rules execute, '
            'TF-IDF never sends commands; not accuracy ground truth or matched causal latency benchmark'
        ),
        'shadow_enabled': shadow_enabled,
        'primary_backend': 'rules_only',
        'pages': results,
        'performance': service_adapter.performance_stats(),
        'resources': {
            'before_model': before,
            'after_model': loaded_memory,
            'after_run': memory_mb(),
            'model_load_ms': load_ms,
            'elapsed_seconds': wall_seconds,
            'python_process_cpu_seconds': cpu_seconds,
            'python_process_cpu_percent_one_core_equivalent': 100 * cpu_seconds / wall_seconds,
        },
        'review_records': len(captured),
        'review_truth': 'unlabelled, potential actions are not confirmed ads or false positives',
        'production_service_touched': False,
        'training_enabled': False,
    }
    (output/'review.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in captured),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'completed':True,'shadow_enabled':shadow_enabled,'performance':report['performance'],'resources':report['resources'],'review_records':len(captured)}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('sites', nargs='*')
    args = parser.parse_args()
    run(args.output.resolve(),not args.baseline_only,args.sites or SITES)


if __name__ == '__main__':
    main()
